import argparse
import copy
from isaaclab.app import AppLauncher

# 1. Launch the app first (Headless = False so we can see the GUI!)
launcher = AppLauncher({"headless": False}) 
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

# 2. Import your specific robot configuration
# IMPORTANT: Replace 'your_extension_name' with your actual package name
from isaac_lab_quadruped.robots.pupper import PUPPER_CFG 

def main():
    # 3. Configure the simulation environment
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Position the camera to look at the center of the world
    sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])

    # 4. Spawn a standard physics ground plane
    sim_utils.spawn_ground_plane("/World/GroundPlane", sim_utils.GroundPlaneCfg())

    # 5. Spawn the robot using your configuration
    robot_cfg = copy.deepcopy(PUPPER_CFG)
    robot_cfg.prim_path = "/World/Robot"  # Tell the simulator where in the scene to put it
    robot = Articulation(cfg=robot_cfg)
    
    # 6. Initialize the simulation and play
    sim.reset()
    print("[INFO] Simulation playing. Press 'ESC' or close the window to exit.")
    
    # 7. Run the physics loop
    while app.is_running():
        # Step the simulation forward
        sim.step()
        # Update the robot's internal state
        robot.update(dt=sim_cfg.dt)

if __name__ == "__main__":
    main()
    # Shut down safely when the window is closed
    app.close()