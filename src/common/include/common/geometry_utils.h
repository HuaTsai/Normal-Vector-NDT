#pragma once
#include <bits/stdc++.h>
#include <geometry_msgs/PoseStamped.h>

geometry_msgs::Pose PoseInterpolate(const geometry_msgs::PoseStamped &p1,
                                    const geometry_msgs::PoseStamped &p2,
                                    const ros::Time &time);

// poses: p0, p1, p2, ..., pt1, (?), pt2, ..., pn
// find interpolate here --------|
geometry_msgs::Pose GetPose(
    const std::vector<geometry_msgs::PoseStamped> &poses,
    const ros::Time &time);
