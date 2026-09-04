# SROS2 / DDS Security Deployment

TLS on the Windows frame bridge does not protect ROS 2 discovery or topic data.
An operational deployment must enable DDS security with an environment-specific
SROS2 keystore. Generated keys and keystores must remain outside this repository.

## Operator workflow

Run these commands on the Ubuntu ROS 2 host, replacing the example path with a
protected location and using the ROS distribution's documented tooling:

```bash
mkdir -p "$HOME/.uav-security"
chmod 700 "$HOME/.uav-security"
ros2 security create_keystore "$HOME/.uav-security/keystore"
ros2 security create_enclave "$HOME/.uav-security/keystore" /uav/uav_windows_tcp_frame_bridge
ros2 security create_enclave "$HOME/.uav-security/keystore" /uav/uav_clean_target_dashboard_v5
ros2 security create_enclave "$HOME/.uav-security/keystore" /uav/uav_analytics_dashboard_v2
ros2 security create_enclave "$HOME/.uav-security/keystore" /uav/uav_tank_type_timeline_dashboard_v1
```

Review and restrict the generated governance/permissions policies to the existing
topics. Then launch each node with its assigned enclave after setting:

```bash
export ROS_SECURITY_KEYSTORE="$HOME/.uav-security/keystore"
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce
ros2 run <package> uav_windows_tcp_frame_bridge --ros-args --enclave /uav/uav_windows_tcp_frame_bridge
```

Use the corresponding enclave for each dashboard. Do not use
`ROS_SECURITY_STRATEGY=Permissive` for a secured deployment. Exact package and
launch names must be confirmed in the live Ubuntu workspace because this
repository contains a Windows mirror rather than the installed ROS package.

Back up the keystore in approved secret storage, restrict filesystem ownership,
rotate identities after compromise, and test that an unenrolled ROS node cannot
discover or subscribe to protected topics.
