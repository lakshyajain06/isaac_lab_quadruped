from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


CURRENT_DIR = Path(__file__).parent
PUPPER_USD_PATH = str((CURRENT_DIR / ".." / "assets" / "pupper_v3" / "pupper_v3.usd").resolve())

PUPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PUPPER_USD_PATH,
        activate_contact_sensors=True
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.13),
        joint_pos={
            ".*_1": 0.0,
            ".*_2": 0.0,
            ".*_3": 0.0
        }
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"], # The regex '.*' applies these gains to ALL joints
            stiffness=5.0,          # The Proportional (P) gain
            damping=0.1,             # The Derivative (D) gain
        ),
    }

)