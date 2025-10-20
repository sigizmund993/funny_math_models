#include "include/auxilary.h"
//variables
Point3D W_b(1,0,0);
float W_d = 1;
Point3D W_r(0,0,-3);
Point3D a_r(1,1,0);
Point3D v_r (3,3,0);
//solution
int main(void)
{
    float maga = (W_b.cross(R_bd)-Point3D(-W_d,0,0).cross(R_db)).mag();
    float a1 = DRIBBLER_MU*(W_b.x*sin(ALPHA)*BALL_R-W_d*sin(ALPHA)*DRIBBLER_R)/maga-cos(ALPHA);
    float a2 = DRIBBLER_MU*(-W_b.x*cos(ALPHA)*BALL_R+W_d*cos(ALPHA)*DRIBBLER_R)/maga+sin(ALPHA);
    float a3 = DRIBBLER_MU*(W_b.y*cos(ALPHA)*BALL_R-W_b.z*sin(ALPHA)*BALL_R)/maga;
    float magb = (W_r.cross(R_bf)-v_r).mag();
    float b1 = 1;
    float b2 = (-W_b.x*BALL_R-v_r.y)/magb*FLOOR_MU;
    float b3 = (W_b.y*BALL_R-v_r.x)/magb*FLOOR_MU;
    float c1 = 0;
    float c2 = 0;
    float c3 = BALL_L-a_r.x;
    float d1 = -g.mag();
    float d2 = (W_r.mag()*W_r.mag())*BALL_L-a_r.y;
    float d3 = 0;
    float beta = (-a1*b2*d3+a1*b3*d2+b1*a2*d3-b1*a3*d2-d1*a2*b3+d1*a3*b2)/(a1*(b2*c3-b3*c2)-b1*(a2*c3-a3*c2)+c1*(a2*b3-a3*b2));
    std::cout<<beta<<std::endl;
    return 0;
}
