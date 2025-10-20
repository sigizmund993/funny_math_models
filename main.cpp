#include <iostream>
#include "include/const.h"
#include "include/field.h"
#include "include/metrics.h"
#include "include/math.h"
#include <vector>
#include "include/time.h"
#include "include/random.h"
#include <cmath>

using namespace std;

int main()
{
    time();
    Point ally_poses[MAX_ROBOT_COUNT], enemy_poses[MAX_ROBOT_COUNT];

    double ally_angles[MAX_ROBOT_COUNT], enemy_angles[MAX_ROBOT_COUNT], metrics;

    Field field(-1);

    FindWay moving;

    for (int i = 0; i < MAX_ROBOT_COUNT; i++) {
        ally_poses[i] = GRAVEYARD_POS;
        ally_angles[i] = 0;
        enemy_poses[i] = GRAVEYARD_POS;
        enemy_angles[i] = 0;
    }

    ally_poses[0] = Point(0, 0);
    ally_poses[1] = Point(-150, 300);
    ally_poses[2] = Point(50, 300);
    ally_poses[3] = Point(-50, 500);
    ally_poses[4] = Point(-50, 700);

    field.update_all(ally_poses, ally_angles, enemy_poses, enemy_angles, Point(0, 0), time());

    moving.reset_config(field, field.allies[0], Point(0, 2000), Point(500, 0), false);

    Trajectory trajectory;
    trajectory.push_back(Waypoint(M_PI / 2.0, 1.0));
    
    moving.evaluate(trajectory);
    return 0;
}