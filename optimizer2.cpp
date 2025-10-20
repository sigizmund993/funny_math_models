#include "include/metrics.h"
#include "include/random.h"
#include "include/time.h"
#include "include/auxilary.h"
#include <algorithm>
#define MAX_MUTATE_ANGLE (M_PI/6) //30 deg
#define GENERATION_SIZE 300
#define ELITE 10 //элитные варвары
#define GENERATIONS 500

struct Individual
{
    public:
    Trajectory trajectory;
    double fitness;
    Individual(Trajectory traj_):trajectory(traj_),fitness(1e10){}
};
typedef vector<Individual> Gen;
void mutate_angle(Trajectory& trajectory)
{
    int idx = random_int(0,trajectory.size()-1);
    double delta_ang = random_double(-MAX_MUTATE_ANGLE,MAX_MUTATE_ANGLE);
    trajectory[idx].acc_angle += delta_ang;
    trajectory[idx].acc_angle = fmod(trajectory[idx].acc_angle + M_PI, 2.0 * M_PI) - M_PI;
}
void mutate_time(Trajectory& trajectory)
{
    int idx = random_int(0,trajectory.size()-1);
    double delta_time = random_double(0.7,1.3);
    trajectory[idx].time *= delta_time;
}
void add_waypoint(Trajectory& trajectory)
{
    int idx = random_int(0,trajectory.size()-1);
    trajectory[idx].time/=2;
    Waypoint new_wp = Waypoint(trajectory[idx].acc_angle+random_double(-MAX_MUTATE_ANGLE,MAX_MUTATE_ANGLE),trajectory[idx].time);
    trajectory.insert(trajectory.begin()+idx+1,new_wp);
}
void remove_waypoint(Trajectory& trajectory)
{
    int idx = random_int(0,trajectory.size()-1);
    trajectory.erase(trajectory.begin()+idx);
}
void mutate(Trajectory& trajectory)
{
    double r = random_double(0,1);//дабл r RR 
    if(r<0.4)//40%
        mutate_angle(trajectory);
    else if (r<0.8)//40%
        mutate_time(trajectory);
    else if (r<0.95)//15%
        add_waypoint(trajectory);
    else//5% 
        remove_waypoint(trajectory);
}
Gen new_generation(Individual parent,int size)
{
    Gen new_gen;
    for(int i = 0;i<size;i++)
    {
        Trajectory new_traj = parent.trajectory;
        mutate(new_traj);
        new_gen.push_back(Individual(new_traj));
    }
    return new_gen;
}
int main()
{
    Field field  = Field(1);
    Robot cur_rbt = field.allies[1];
    Point tgt_pos = Point(2000,2000);
    Point tgt_vel = Point(-1000,0);
    double start_time = time();
    FindWay estimator;
    estimator.reset_config(field,cur_rbt,tgt_pos,tgt_vel);
    Trajectory start_traj = {Waypoint(0,10),Waypoint(20,20)};//bangbang
    Individual first_man = Individual(start_traj);
    first_man.fitness = estimator.estimate(first_man.trajectory);
    Gen new_gen = new_generation(first_man,GENERATION_SIZE);
    for(Individual man:new_gen)
        man.fitness = estimator.estimate(man.trajectory);
    sort(new_gen.begin(), new_gen.end(), [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
    Gen best_individuals(new_gen.begin(),new_gen.begin()+ELITE);
    for(int i = 0;i<GENERATIONS;i++)
    {
        new_gen.clear();
        for(Individual men:best_individuals)//men men men
        {
            Gen new_smol_gen = new_generation(men,GENERATION_SIZE/ELITE);
            new_gen.insert(new_gen.end(),new_smol_gen.begin(),new_smol_gen.end());
        }
        for(Individual man:new_gen)
            man.fitness = estimator.estimate(man.trajectory);
        sort(new_gen.begin(), new_gen.end(), [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });
        Gen best_individuals(new_gen.begin(),new_gen.begin()+ELITE);
    }
    double end_time = time();
    printf("%f",end_time-start_time);
    return 0;
}
