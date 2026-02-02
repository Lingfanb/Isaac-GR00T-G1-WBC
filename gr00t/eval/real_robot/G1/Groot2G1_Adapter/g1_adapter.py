"""GR00T to G1 Adapter - Observation and Action conversion

Converts G1 robot observations to GR00T VLA input format,
and decodes GR00T action outputs to G1 motor commands.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import yaml


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "g1_policy_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.

    GR00T Policy Server expects:
        obs: (batch=1, time=1, ...)
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar → [scalar]
    return obs


class G1Adapter:
    """
    Adapter between G1 robot and GR00T VLA format.
    
    Responsible for:
        • Packaging camera frames as obs["video"]
        • Building obs["state"] for dual arms + hands
        • Adding language instruction
        • Adding batch/time dimensions
        • Decoding model action chunks into real robot commands
    
    Configuration loaded from g1_policy_config.yaml.
    """

    def __init__(
        self, 
        policy_client: Any,
        config_path: Optional[str] = None,
    ):
        """
        Initialize G1 adapter.
        
        Args:
            policy_client: PolicyClient instance from gr00t.policy.server_client
            config_path: Path to g1_policy_config.yaml. If None, uses default.
        """
        self.policy = policy_client
        self.config = load_config(config_path)
        
        # Parse configuration
        self._parse_obs_config()
        self._parse_action_config()
    
    def _parse_obs_config(self):
        """Parse observation configuration from YAML."""
        obs_cfg = self.config.get("observation", {})
        
        # State components
        state_cfg = obs_cfg.get("state", {})
        self.state_keys = []
        self.state_indices = {}
        
        for key, cfg in state_cfg.items():
            if cfg.get("enabled", True):
                self.state_keys.append(key)
                if "indices" in cfg:
                    self.state_indices[key] = cfg["indices"]
        
        # Camera keys
        video_cfg = obs_cfg.get("video", {})
        self.camera_keys = [
            key for key, cfg in video_cfg.items() 
            if cfg.get("enabled", True)
        ]
        self.camera_source_map = {
            key: cfg.get("source_key", key) 
            for key, cfg in video_cfg.items()
        }
        
        # Language config
        lang_cfg = obs_cfg.get("language", {})
        self.lang_key = lang_cfg.get("key", "annotation.human.task_description")
        self.lang_source = lang_cfg.get("source_key", "lang")
    
    def _parse_action_config(self):
        """Parse action configuration from YAML."""
        action_cfg = self.config.get("action", {})
        
        self.action_keys = []
        self.action_target_indices = {}
        
        for key, cfg in action_cfg.items():
            if cfg.get("enabled", True):
                self.action_keys.append(key)
                if "target_indices" in cfg:
                    self.action_target_indices[key] = cfg["target_indices"]

    # -------------------------------------------------------------------------
    # Observation → Model Input
    # -------------------------------------------------------------------------
    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        """
        Convert raw G1 observation dict into the structured GR00T VLA input.
        
        Expected obs format from G1RobotInterface.get_observation():
            {
                "motor_positions": np.ndarray (29,),
                "left_arm_positions": np.ndarray (7,),
                "right_arm_positions": np.ndarray (7,),
                "left_hand_positions": np.ndarray (7,),  # if hands enabled
                "right_hand_positions": np.ndarray (7,), # if hands enabled
                "head_camera": np.ndarray (H, W, 3),     # if camera enabled
                "lang": str,
            }
        """
        model_obs = {}

        # (1) Cameras
        model_obs["video"] = {}
        for key in self.camera_keys:
            source_key = self.camera_source_map.get(key, key)
            if source_key in obs:
                model_obs["video"][key] = obs[source_key]
            elif f"{key}_camera" in obs:
                model_obs["video"][key] = obs[f"{key}_camera"]

        # (2) Robot state
        model_obs["state"] = {}
        for key in self.state_keys:
            # Try multiple possible source keys
            possible_keys = [
                f"{key}_positions",           # e.g., "left_arm_positions"
                f"{key.replace('_', '')}",    # e.g., "leftarm"
                key,                          # e.g., "left_arm"
            ]
            
            value = None
            for k in possible_keys:
                if k in obs:
                    value = obs[k]
                    break
            
            # If still not found, extract from motor_positions using indices
            if value is None and key in self.state_indices:
                motor_pos = obs.get("motor_positions", np.zeros(29))
                indices = self.state_indices[key]
                value = motor_pos[indices]
            
            if value is None:
                value = np.zeros(7)  # Default
                
            model_obs["state"][key] = np.asarray(value, dtype=np.float32)

        # (3) Language instruction
        lang = obs.get(self.lang_source, obs.get("language", ""))
        model_obs["language"] = {self.lang_key: lang}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        
        return model_obs

    # -------------------------------------------------------------------------
    # Model Action Chunk → Robot Motor Commands
    # -------------------------------------------------------------------------
    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, np.ndarray]:
        """
        Decode GR00T action chunk to G1 motor commands for timestep t.
        
        Expected chunk format from GR00T:
            chunk["left_arm"]:  (B, T, 7)
            chunk["right_arm"]: (B, T, 7)
            chunk["left_hand"]:  (B, T, 7)  # if hands enabled
            chunk["right_hand"]: (B, T, 7)  # if hands enabled
        
        Returns:
            Dict with keys matching self.action_keys
        """
        action = {}
        
        for key in self.action_keys:
            if key in chunk:
                action[key] = np.array(chunk[key][0][t], dtype=np.float32)
        
        return action

    def get_action(self, obs: Dict) -> List[Dict[str, np.ndarray]]:
        """
        Get action sequence from policy.
        
        Returns a list of action dicts (one per model timestep).
        """
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        # Determine horizon from first available key
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]
    
    def action_to_robot_command(self, action: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Convert adapter action to G1RobotInterface command format.
        
        This can be used directly with robot.send_action().
        """
        cmd = {}
        
        if "left_arm" in action:
            cmd["left_arm"] = action["left_arm"]
        if "right_arm" in action:
            cmd["right_arm"] = action["right_arm"]
        
        # Combine arm positions if needed
        if "left_arm" in action and "right_arm" in action:
            cmd["arm_positions"] = np.concatenate([
                action["left_arm"], 
                action["right_arm"]
            ])
        
        # Hands are handled separately via Dex3HandInterface
        if "left_hand" in action:
            cmd["left_hand"] = action["left_hand"]
        if "right_hand" in action:
            cmd["right_hand"] = action["right_hand"]
        
        return cmd
    
    @property
    def action_horizon(self) -> int:
        """Get configured action horizon."""
        return self.config.get("policy", {}).get("action_horizon", 8)
    
    @property
    def action_frequency(self) -> float:
        """Get configured action frequency (Hz)."""
        return self.config.get("control", {}).get("action_frequency", 30)
