hello world
Holos Team 1 Project

Description: This project explores using Apple Vision Pro as a real-time teleoperation interface for robotic manipulation and demonstration collection. Building on the existing VisionProTeleop framework, we aim to map Vision Pro hand-tracking data to a simulated Shadow Hand in MuJoCo, enabling users to control robotic movements while recording robot states, actions, object poses, and task outcomes. We plan to extend the system with robotic arms for object manipulation and collect demonstrations of household tasks to support robot policy training.

## Emma development baseline

Start with [Team setup and contribution instructions](shadow-hand-teleop/TEAM_SETUP.md).
The `Emma` branch includes experimental arm-equipped scenes and known limitations;
it is not a release-quality controller. Create your own named branch from this
baseline, coordinate an area of work, and contribute through a PR. Existing
mock export and HDF5/Holos replay tools are preserved. The live arm viewer does
not yet save robot demonstrations; movable shoulder bases and BunnyVisionPro
integration remain open work.
