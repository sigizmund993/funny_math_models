#pragma once

#include "const.h"
#include "auxilary.h"
#include <cmath>
#include <vector>

using namespace std;

class Ball
{
protected:
    Point pos, vel, acc;
    double R, last_update;

public:
    Ball(Point pos_, double R_ = BALL_R) : pos(pos_.x, pos_.y), R(R_) {}
    Ball() : pos(0.0, 0.0), R(BALL_R) {}
    void update(Point new_pos, double t)
    {
        static double dt;
        dt = t - last_update;
        acc = ((new_pos - pos) / dt - vel) / dt;
        vel = (new_pos - pos) / dt;
        pos = new_pos;
    }
    // ball params
    void set_r(double R_)
    {
        R = R_;
    }
    double get_r() const
    {
        return R;
    }
    Point get_pos() const
    {
        return pos;
    }
    Point get_vel() const
    {
        return vel;
    }
    Point get_acc() const
    {
        return acc;
    }
};

ostream &operator<<(ostream &os, const Ball &ball)
{
    os << "pos: " << ball.get_pos() << " vel: " << ball.get_vel() << " acc: "  << ball.get_acc();
    return os;
}

class Robot : public Ball
{
private:
    int r_id, color, exclude_active;
    bool is_used = false;
    double lifetime, angle, angle_vel, angle_acc;

public:
    Robot() : Ball(GRAVEYARD_POS, ROBOT_R), angle(0.0) {}
    void update(Point new_pos, double new_angle, double t)
    {
        static double dt;
        dt = t - last_update;
        acc = ((new_pos - pos) / dt - vel) / dt;
        vel = (new_pos - pos) / dt;
        pos = new_pos;
        angle_acc = ((new_angle - angle) / dt - angle_vel) / dt;
        angle_vel = (new_angle - angle) / dt;
        angle = new_angle;
        // cout << r_id << ", " << color << ", " << pos.x << ", " << pos.y << endl;
        if ((pos == GRAVEYARD_POS) ^ is_used)
        {
            // cout << "aaa" << endl;
            lifetime = t;
        }
        else {
            // cout << "bbb" << endl;
        }
        if (t - lifetime > TIME_TO_BORN)
        {
            is_used = true;
            vel = Point(0, 0);
            acc = Point(0, 0);
            angle_vel = 0;
            angle_acc = 0;
        }
        else if (t - lifetime > TIME_TO_DIE)
        {
            is_used = false;
        }
        last_update = t;
    }
    // rbt params
    bool get_used() const
    {
        return is_used;
    }
    void set_id(int r_id_)
    {
        r_id = r_id_;
    }
    int get_id() const
    {
        return r_id;
    }
    void set_color(int color_)
    {
        color = color_;
    }
    int get_color() const
    {
        return color;
    }
    double get_angle() const
    {
        return angle;
    }
    double get_angle_vel() const
    {
        return angle_vel;
    }
    double get_angle_acc() const
    {
        return angle_acc;
    }
    bool operator==(Robot other) const
    {
        return r_id == other.get_id() && color == other.get_color();
    }
    bool operator!=(Robot other) const
    {
        return !(*this == other);
    }
};

ostream &operator<<(ostream &os, const Robot &rbt)
{
    os << "id: " << rbt.get_id() << " color: " << rbt.get_color() << "pos: " << rbt.get_pos() << " vel: " << rbt.get_vel() << " acc: "  << rbt.get_acc();
    return os;
}

struct Goal
{
    Point center, up, down, frw_up, frw_down, frw_center, center_up, center_down;
    Point hull[5], big_hull[5];
    Goal(int polarity) : center(FIELD_DX / 2.0 * polarity, 0),
                         up(FIELD_DX / 2.0 * polarity, ZONE_DY / 2.0),
                         down(FIELD_DX / 2.0 * polarity, -ZONE_DY / 2.0),
                         frw_up(FIELD_DX / 2.0 * polarity - ZONE_DX * polarity, ZONE_DY / 2.0),
                         frw_down(FIELD_DX / 2.0 * polarity - ZONE_DX * polarity, -ZONE_DY / 2.0),
                         frw_center(FIELD_DX / 2.0 * polarity - ZONE_DX * polarity, 0),
                         center_up(FIELD_DX / 2.0 * polarity, GOAL_DY * polarity / 2.0),
                         center_down(FIELD_DX / 2.0 * polarity, -GOAL_DY * polarity / 2.0)
    {
        hull[0] = this->center_up;
        hull[1] = this->frw_up;
        hull[2] = this->frw_down;
        hull[3] = this->center_down;
        hull[4] = Point(INF * polarity, 0);
        big_hull[0] = hull[0] + Point(0, ROBOT_R * polarity);
        big_hull[1] = hull[1] + Point(-ROBOT_R * polarity, ROBOT_R * polarity);
        big_hull[2] = hull[2] + Point(-ROBOT_R * polarity, -ROBOT_R * polarity);
        big_hull[3] = hull[3] + Point(0, -ROBOT_R * polarity);
        big_hull[4] = hull[4];
    }
};

class Field
{
public:
    Goal ally_goal, enemy_goal;
    Point hull[4];
    Robot allies[MAX_ROBOT_COUNT], enemies[MAX_ROBOT_COUNT];
    vector<Robot> active_allies = {}, active_enemies = {};
    Ball ball;
    Field(int polarity) : ally_goal(polarity), enemy_goal(-polarity)
    {
        active_allies.reserve(MAX_ROBOT_COUNT);
        active_enemies.reserve(MAX_ROBOT_COUNT);
        for (int i = 0; i < MAX_ROBOT_COUNT; i++)
        {
            allies[i].set_id(i);
            allies[i].set_color(COLOR);
            enemies[i].set_id(i);
            enemies[i].set_color(YELLOW + BLUE - COLOR);
        }
        hull[0] = Point(FIELD_DX, FIELD_DY);
        hull[1] = Point(FIELD_DX, -FIELD_DY);
        hull[2] = Point(-FIELD_DX, -FIELD_DY);
        hull[3] = Point(-FIELD_DX, FIELD_DY);
    }

    void update_active_robots()
    {
        static int i;
        active_allies.clear();
        active_enemies.clear();
        for (i = 0; i < MAX_ROBOT_COUNT; i++) {
            if (allies[i].get_used())
            {
                active_allies.push_back(allies[i]);
            }
            if (enemies[i].get_used())
            {
                active_enemies.push_back(enemies[i]);
            }
        }
    }

    void update_all(Point *ally_robots_poses, double *ally_robot_angles, Point *enemy_robots_poses, double *enemy_robots_angles, Point ball_pos, double t)
    {
        static int i;
        for (i = 0; i < MAX_ROBOT_COUNT; i++)
        {
            allies[i].update(ally_robots_poses[i], ally_robot_angles[i], t);
            enemies[i].update(enemy_robots_poses[i], enemy_robots_angles[i], t);
        }
        ball.update(ball_pos, t);
        update_active_robots();
    }
};
