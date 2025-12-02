#pragma once
#include "cuda_auxilary.cuh"
#include "const.cuh"

#define MAX_TRAJ_STEPS 64

struct Waypoint
{
    float acc_angle, time;
    __host__ __device__ Waypoint() : acc_angle(0), time(0) {}
    __host__ __device__ Waypoint(float acc_angle_, float time_) : acc_angle(acc_angle_), time(time_) {}
};

struct Trajectory
{
    Waypoint steps[MAX_TRAJ_STEPS];
    int length;

    __host__ __device__ void reset() { length = 0; }

    __host__ __device__ void append(float angle, float time)
    {
        if (length < MAX_TRAJ_STEPS)
            steps[length++] = Waypoint(angle, time);
    }

    // удаляет элемент по индексу idx (сдвигает все элементы после него)
    __host__ __device__ void remove(int idx)
    {
        if (idx < 0 || idx >= length) return;
        for (int i = idx; i < length - 1; i++)
        {
            steps[i] = steps[i + 1];
        }
        length--;
    }

    // вставляет элемент на позицию idx (сдвигает элементы вправо)
    __host__ __device__ void insert(int idx, float angle, float time)
    {
        if (length >= MAX_TRAJ_STEPS) return;
        if (idx < 0) idx = 0;
        if (idx > length) idx = length;

        for (int i = length; i > idx; i--)
        {
            steps[i] = steps[i - 1];
        }
        steps[idx] = Waypoint(angle, time);
        length++;
    }

    // доступ к элементу по индексу (можно менять)
    __host__ __device__ Waypoint& operator[](int idx)
    {
        // без проверки на GPU, на CPU можно добавить assert
        return steps[idx];
    }

    __host__ __device__ const Waypoint& operator[](int idx) const
    {
        return steps[idx];
    }
};
struct Obstacle
{
    Point pos;
    float r;
    __host__ __device__ Obstacle() : pos(0,0), r(0) {}
    __host__ __device__ Obstacle(Point _pos, float _r) : pos(_pos), r(_r) {}

    
};
struct Obstacle_list
{
    Obstacle obs[MAX_ROBOT_COUNT];
    int length;

    __host__ __device__ void reset() { length = 0; }

    __host__ __device__ void append(Point pos, float r)
    {
        if (length < MAX_ROBOT_COUNT)
            obs[length++] = Obstacle(pos, r);
    }

    // удаляет элемент по индексу idx (сдвигает все элементы после него)
    __host__ __device__ void remove(int idx)
    {
        if (idx < 0 || idx >= length) return;
        for (int i = idx; i < length - 1; i++)
        {
            obs[i] = obs[i + 1];
        }
        length--;
    }

    // вставляет элемент на позицию idx (сдвигает элементы вправо)
    __host__ __device__ void insert(int idx, Point pos,float r)
    {
        if (length >= MAX_TRAJ_STEPS) return;
        if (idx < 0) idx = 0;
        if (idx > length) idx = length;

        for (int i = length; i > idx; i--)
        {
            obs[i] = obs[i - 1];
        }
        obs[idx] = Obstacle(pos,r);
        length++;
    }

    // доступ к элементу по индексу (можно менять)
    __host__ __device__ Obstacle& operator[](int idx)
    {
        // без проверки на GPU, на CPU можно добавить assert
        return obs[idx];
    }

    __host__ __device__ const Obstacle& operator[](int idx) const
    {
        return obs[idx];
    }
};

struct estimate_data
{
    Point r0,v0,re,ve;
    float mint;//time of optimal trajectory that ignores obstacles (bang bang or just max speed,hzhz)
    Obstacle_list obstacles;
    __host__ __device__ estimate_data(Point _r0,Point _v0,Point _re,Point _ve):r0(_r0),v0(_v0),re(_re),ve(_ve){}
};

float estimate_rv(estimate_data data, Trajectory traj)
{
    int i;
    Point r = data.r0;
    Point vel = data.v0;
    Point acc = Point(MAX_ACC,0);
    float time;
    float timesat;
    float cosalpha;
    for(i = 0;i<traj.length;i++)
    {
        acc = Point(MAX_ACC,0).rotate(traj[i].acc_angle);
        time = traj[i].time;
        
        if((vel+acc*time).mag()>MAX_SPEED)
        {
            cosalpha = cosf(get_angle_between_points(Point(0,0),vel,vel+acc));
            timesat = (cosalpha*vel.mag()+sqrtf(cosalpha*cosalpha*vel.mag2()+MAX_SPEED*MAX_SPEED-vel.mag2()))/MAX_ACC;
            r += vel*timesat + acc*timesat*timesat/2;
            vel += acc*timesat;
            r+= vel*(time-timesat);
        }   
        else
        {
            r += vel*time+acc*time*time/2;
            vel+=acc*time;
        }
        
        
    }
    float velerr = (vel-data.ve).mag();
    float rerr = (r-data.re).mag();
    return velerr+rerr;
}
float estimate_time(estimate_data data, Trajectory traj)
{
    int i;
    float total_time;
    for(i = 0;i<traj.length;i++)
        total_time += traj[i].time;
    return total_time-data.mint;
}
float estimate_collisions(estimate_data data, Trajectory traj)
{
    
    float est = 0;
    return est;
}