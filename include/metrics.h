#pragma once

#include <iostream>
#include "field.h"
#include "const.h"
#include "auxilary.h"
#include <complex>
#include <vector>
#include "quicksort.h"

using namespace std;

struct Waypoint
{
    double acc_angle, time;
    Waypoint(double acc_angle_, double time_) : acc_angle(acc_angle_) , time(time_){}
};

using Trajectory = vector<Waypoint>;

class FindWay
{
private:
    Object obj[2 * MAX_ROBOT_COUNT + 1];
    Point r_mv[2 * MAX_SOLVER_DEGREE + 1], v_mv[2 * MAX_SOLVER_DEGREE + 1], a_mv[2 * MAX_SOLVER_DEGREE], target_pos, target_vel, m_center[2 * MAX_ROBOT_COUNT + 1];
    double t_move[2 * MAX_SOLVER_DEGREE + 1];
    vector<double> roots[2 * MAX_ROBOT_COUNT + 1], grp_rts[2 * MAX_ROBOT_COUNT + 1];
    int n_move, n_objects, n_groups, i, j, k, group[2 * MAX_ROBOT_COUNT + 1];
    int get_t_idx(double t)
    {
        if (t < 0 || t > t_move[n_move])
        {
            return -1;
        }
        if (t == 0)
        {
            return 0;
        }
        static int idx;
        for (idx = 1; t_move[idx] < t; idx++)
            ;
        return idx - 1;
    }

    Point get_current_pos(double t, int idx_ = -1)
    {
        static double dt;
        static int idx;
        if (idx_ >= 0 && idx_ < n_move)
        {
            idx = idx_;
        }
        else
        {
            idx = get_t_idx(t);
        }
        if (idx == -1)
        {
            return Point(0, 0, true);
        }
        dt = t - t_move[idx];
        return r_mv[idx] + v_mv[idx] * dt + a_mv[idx] * dt * dt / 2.0;
    }

    Point get_current_vel(double t, int idx_ = -1)
    {
        static int idx;
        if (idx_ >= 0 && idx_ < n_move)
        {
            idx = idx_;
        }
        else
        {
            idx = get_t_idx(t);
        }
        if (idx == -1)
        {
            return Point(0, 0, true);
        }
        return v_mv[idx] + a_mv[idx] * (t - t_move[idx]);
    }

    Point get_current_acc(double t, int idx_ = -1)
    {
        static int idx;
        if (idx_ >= 0 && idx_ < n_move)
        {
            idx = idx_;
        }
        else
        {
            idx = get_t_idx(t);
        }
        if (idx == -1)
        {
            return Point(0, 0, true);
        }
        return a_mv[idx];
    }

    void reset_moves()
    {
        n_move = 0;
        t_move[0] = 0;
        for (i = 0; i < 2 * MAX_ROBOT_COUNT + 1; i++)
        {
            roots[i] = vector<double>();
            roots[i].push_back(0.0);
            grp_rts[i] = vector<double>();
        }
    }

    void append_segment(double angle, double t)
    {
        static double acc_time, min_t, scal;
        static Point r, v, a, rp, acc;
        if (t == 0)
            return;
        a_mv[n_move] = Point(MAX_ACC, 0).rotate(angle);
        scal = a_mv[n_move].scalar(v_mv[n_move]);
        acc_time = (-scal + sqrt(scal * scal + 4.0 * MAX_ACC * MAX_ACC * (MAX_SPEED * MAX_SPEED - v_mv[n_move].mag2()))) / (2 * MAX_ACC * MAX_ACC);
        if (acc_time > EPSILON)
        {
            min_t = fmin(acc_time, t);
            t_move[n_move + 1] = t_move[n_move] + min_t;
            v_mv[n_move + 1] = v_mv[n_move] + a_mv[n_move] * min_t;
            r_mv[n_move + 1] = r_mv[n_move] + (v_mv[n_move] + v_mv[n_move + 1]) * min_t / 2.0;
            n_move++;
        }
        if (t > acc_time)
        {
            a_mv[n_move] = Point(0, 0);
            t_move[n_move + 1] = t_move[n_move] + t - acc_time;
            v_mv[n_move + 1] = v_mv[n_move];
            r_mv[n_move + 1] = r_mv[n_move] + v_mv[n_move + 1] * (t - acc_time);
            n_move++;
        }
    }

    void find_roots()
    {
        static int poses[3], size, n_rts;
        static double average_dist, ak, bk, ck, dk, ek, real_rt;
        static bool gone_in;
        static complex<double> complex_roots[4];
        for (i = 0; i < n_move; i++)
        {
            for (j = 0; j < n_objects; j++)
            {
                ek = (r_mv[i] - obj[j].c).mag2() - SQUARE(ROBOT_R + obj[j].r);
                dk = 2.0 * v_mv[i].scalar(r_mv[i] - obj[j].c);      
                ck = r_mv[i].scalar(a_mv[i]) + v_mv[i].mag2() - a_mv[i].scalar(obj[j].c);
                bk = v_mv[i].scalar(a_mv[i]);
                ak = a_mv[i].mag2() / 4.0;
                n_rts = solve_four(ak, bk, ck, dk, ek, complex_roots);
                for (k = 0; k < n_rts; k++)
                {
                    if (abs(imag(complex_roots[k])) < EPSILON)
                    {
                        real_rt = real(complex_roots[k]);
                        if (real_rt >= 0 && real_rt < t_move[i + 1] - t_move[i])
                        {
                            roots[j].push_back(real_rt + t_move[i]);
                        }
                    }
                }
            }
        }
        for (i = 0; i < n_objects; i++)
        {
            size = roots[i].size() + 1;
            if (size > 2)
            {
                abs_sort(roots[i], 1, size - 2);
                roots[i].push_back(t_move[n_move]);
                k = -1;
                for (j = 0; j < 3; j++)
                {
                    for (k++; k < size - 1 && roots[i][k + 1] - roots[i][k] < EPSILON; k++)
                        ;
                    poses[j] = k;
                }
                if (poses[1] == size)
                {
                    continue;
                }
                if (poses[0] != 0 && (get_current_pos((roots[i][poses[0]] + roots[i][poses[1]]) / 2.0) - obj[i].c).mag() < ROBOT_R + obj[i].r)
                {
                    grp_rts[group[i]].push_back(roots[i][poses[0]]);
                    gone_in = true;
                }
                else
                {
                    gone_in = false;
                }
                if (poses[2] < size)
                {
                    average_dist = (get_current_pos((roots[i][poses[0]] + roots[i][poses[1]]) / 2.0) - obj[i].c).mag() - ROBOT_R - obj[i].r;
                    if (average_dist * ((get_current_pos((roots[i][poses[1]] + roots[i][poses[2]]) / 2.0) - obj[i].c).mag() - ROBOT_R - obj[i].r) < 0)
                    {
                        if (average_dist < 0)
                        {
                            if (!gone_in)
                            {
                                grp_rts[group[i]].push_back(0);
                            }
                            grp_rts[group[i]].push_back(-roots[i][poses[1]]);
                        }
                        else
                        {
                            grp_rts[group[i]].push_back(roots[i][poses[1]]);
                        }
                    }
                    gone_in = true;
                }
                for (j = poses[2]; j < size - 1;)
                {
                    for (j++; j < size - 1 && roots[i][j + 1] - roots[i][j] < EPSILON; j++)
                        ;
                    poses[0] = poses[1];
                    poses[1] = poses[2];
                    poses[2] = j;
                    average_dist = (get_current_pos((roots[i][poses[0]] + roots[i][poses[1]]) / 2.0) - obj[i].c).mag() - ROBOT_R - obj[i].r;
                    if (average_dist * ((get_current_pos((roots[i][poses[1]] + roots[i][poses[2]]) / 2.0) - obj[i].c).mag() - ROBOT_R - obj[i].r) < 0)
                    {
                        if (average_dist < 0)
                        {
                            grp_rts[group[i]].push_back(-roots[i][poses[1]]);
                        }
                        else
                        {
                            grp_rts[group[i]].push_back(roots[i][poses[1]]);
                        }
                    }
                }
                size = grp_rts[group[i]].size() - 1;
                if (gone_in && grp_rts[group[i]][size] >= 0)
                {
                    grp_rts[group[i]].push_back(-t_move[n_move]);
                }
            }
        }
        for (i = 0; i < n_groups; i++)
        {
            abs_sort(grp_rts[i], 0, grp_rts[i].size() - 1);
        }
    }

    double solve_metrics()
    {
        static int enter_idx, group_score, mv_idx[3], grp_size;
        static Point closest, start_vals[3];
        static double minVal, val, metrics[4];
        metrics[0] = (r_mv[n_move] - target_pos).mag() / MAX_SPEED;
        if (metrics[0] < 0.05)
        {
            metrics[0] = 0;
        }
        metrics[1] = (v_mv[n_move] - target_vel).mag() / MAX_ACC;
        if (metrics[1] < 0.05)
        {
            metrics[1] = 0;
        }
        metrics[2] = 0;
        for (i = 0; i < n_groups; i++)
        {
            enter_idx = 0;
            group_score = 1;
            grp_size = grp_rts[i].size();
            for (j = 1; j < grp_size; j++)
            {
                if (grp_rts[i][j] >= 0)
                {
                    group_score++;
                }
                else
                {
                    group_score--;
                }
                if (group_score == 0)
                {
                    mv_idx[0] = get_t_idx(grp_rts[i][enter_idx]) + 1;
                    mv_idx[1] = get_t_idx(-grp_rts[i][j]);
                    start_vals[0] = get_current_pos(grp_rts[i][enter_idx], mv_idx[0] - 1);
                    start_vals[1] = get_current_vel(grp_rts[i][enter_idx], mv_idx[0] - 1);
                    start_vals[2] = get_current_acc(grp_rts[i][enter_idx], mv_idx[0] - 1);
                    if (mv_idx[1] < mv_idx[0])
                    {
                        closest = closest_point_on_parabola(m_center[i], start_vals[0], start_vals[1], start_vals[2], 0, -grp_rts[i][j] - grp_rts[i][enter_idx]);
                        minVal = (closest - m_center[i]).mag();
                    }
                    else
                    {
                        k = mv_idx[0];
                        closest = closest_point_on_parabola(m_center[i], start_vals[0], start_vals[1], start_vals[2], 0, t_move[mv_idx[0]] - grp_rts[i][enter_idx]);
                        minVal = (closest - m_center[i]).mag();
                        for (; k < mv_idx[1]; k++)
                        {
                            closest = closest_point_on_parabola(m_center[i], r_mv[k], v_mv[k], a_mv[k], 0, t_move[k + 1] - t_move[k]);
                            val = (closest - m_center[i]).mag();
                            if (val < minVal)
                            {
                                minVal = val;
                            }
                        }
                        closest = closest_point_on_parabola(m_center[i], r_mv[k], v_mv[k], a_mv[k], 0, -grp_rts[i][j] - t_move[k]);
                        val = (closest - m_center[i]).mag();
                        if (val < minVal)
                        {
                            minVal = val;
                        }
                    }
                    cout << minVal << " aaa" << endl;
                    metrics[2] += pow(POW_BASE, -minVal);
                    enter_idx = j + 1;
                }
            }
        }
        metrics[3] = t_move[n_move];//савелий хуйланчик
        // for (i = 0; i < 4; i++) {
        //     cout << metrics[i] << endl;
        // }
        return metrics[0] * DIST_K + metrics[1] * VEL_K + metrics[2] * OBSTACLE_K + metrics[3];
    }

public:
    void reset_config(const Field &field, const Robot &cur_rbt, Point target_pos_, Point target_vel_, bool ball_collision = true)
    {
        static int n_group;
        target_pos = target_pos_;
        target_vel = target_vel_;
        r_mv[0] = cur_rbt.get_pos();
        v_mv[0] = cur_rbt.get_vel();
        n_objects = 0;
        for (const Robot &rbt : field.active_allies)
        {
            if (rbt != cur_rbt && (r_mv[0] - rbt.get_pos()).mag() >= 2.0 * ROBOT_R && (target_pos - rbt.get_pos()).mag() >= 2.0 * ROBOT_R)
            {
                // cout << rbt.get_id() << ", " << rbt.get_color() << endl;
                obj[n_objects] = Object(rbt.get_pos(), rbt.get_r());
                n_objects++;
            }
        }
        for (const Robot &rbt : field.active_enemies)
        {
            if (rbt != cur_rbt && (r_mv[0] - rbt.get_pos()).mag() >= 2.0 * ROBOT_R && (target_pos - rbt.get_pos()).mag() >= 2.0 * ROBOT_R)
            {
                // cout << rbt.get_id() << ", " << rbt.get_color() << endl;
                obj[n_objects] = Object(rbt.get_pos(), rbt.get_r());
                n_objects++;
            }
        }
        if (ball_collision && (r_mv[0] - field.ball.get_pos()).mag() >= ROBOT_R + BALL_R && (target_pos - field.ball.get_pos()).mag() >= ROBOT_R + BALL_R)
        {
            obj[n_objects] = Object(field.ball.get_pos(), field.ball.get_r());
            n_objects++;
        }
        n_groups = 0;
        for (i = 0; i < n_objects; i++)
        {
            group[i] = -1;
            for (j = i - 1; j >= 0; j--)
            {
                if ((obj[i].c - obj[j].c).mag() < obj[i].r + obj[j].r + 2 * ROBOT_R)
                {
                    group[i] = group[j];
                }
            }
            if (group[i] == -1)
            {
                group[i] = n_groups;
                n_groups++;
            }
        }
        for (i = 0; i < n_groups; i++)
        {
            n_group = 0;
            for (j = 0; j < n_objects; j++)
            {
                if (group[j] == i)
                {
                    m_center[i] += obj[j].c;
                    n_group++;
                }
            }
            m_center[i] /= n_group;
        }
    }

    double estimate(Trajectory &trajectory)
    {
        reset_moves();
        for (const Waypoint& wp : trajectory) {
            append_segment(wp.acc_angle, wp.time);
        }
        find_roots();
        return solve_metrics();
    }
};