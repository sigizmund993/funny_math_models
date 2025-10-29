#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include "const.h"

using namespace std;

#define SQUARE(x) ((x) * (x))
// class Point3D {
// public: 
//     double x, y, z;

//     // Конструкторы
//     Point3D() : x(0), y(0), z(0) {}
//     Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

//     // Операторы сложения и вычитания
//     Point3D operator+(const Point3D& other) const {
//         return Point3D(x + other.x, y + other.y, z + other.z);
//     }
//     Point3D operator-(const Point3D& other) const {
//         return Point3D(x - other.x, y - other.y, z - other.z);
//     }

//     // Операторы умножения и деления на скаляр
//     Point3D operator*(double scalar) const {
//         return Point3D(x * scalar, y * scalar, z * scalar);
//     }
//     Point3D operator/(double scalar) const {
//         // Добавлена проверка на деление на ноль
//         if (scalar == 0.0) {
//             // В зависимости от задачи, можно выбросить исключение или вернуть нулевой вектор
//             return Point3D(0, 0, 0); 
//         }
//         return Point3D(x / scalar, y / scalar, z / scalar);
//     }

//     // Составные операторы
//     Point3D& operator+=(const Point3D& other) {
//         x += other.x;
//         y += other.y;
//         z += other.z;
//         return *this;
//     }
//     Point3D& operator-=(const Point3D& other) {
//         x -= other.x;
//         y -= other.y;
//         z -= other.z;
//         return *this;
//     }
//     Point3D& operator*=(double scalar) {
//         x *= scalar;
//         y *= scalar;
//         z *= scalar;
//         return *this;
//     }
//     Point3D& operator/=(double scalar) {
//         // Добавлена проверка на деление на ноль
//         if (scalar == 0.0) {
//             // Обработка ошибки
//             return *this;
//         }
//         x /= scalar;
//         y /= scalar;
//         z /= scalar;
//         return *this;
//     }

//     // Операторы сравнения
//     bool operator==(const Point3D& other) const {
//         return x == other.x && y == other.y && z == other.z;
//     }
//     bool operator!=(const Point3D& other) const {
//         return !(*this == other);
//     }

//     // Скалярное произведение (dot product)
//     double dot(const Point3D& other) const {
//         return x * other.x + y * other.y + z * other.z;
//     }

//     // Векторное произведение (cross product)
//     Point3D cross(const Point3D& other) const {
//         return Point3D(
//             y * other.z - z * other.y,
//             z * other.x - x * other.z,
//             x * other.y - y * other.x
//         );
//     }

//     // Вычисление длины вектора (magnitude)
//     double mag() const {
//         return std::sqrt(x * x + y * y + z * z);
//     }
    
//     // Возвращает нормализованный вектор (длина равна 1)
//     Point3D unity() const {
//         double len = mag();
//         return len > 0.0 ? Point3D(x / len, y / len, z / len) : Point3D(0.0, 0.0, 0.0);
//     }
// };

// // Функция для вывода вектора в поток
// std::ostream& operator<<(std::ostream& os, const Point3D& p) {
//     os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
//     return os;
// }
class Point
{
public:
    double x, y;
    bool is_none;
    Point() : x(0), y(0), is_none(false) {}
    Point(double x_, double y_, bool is_none_ = false) : x(x_), y(y_), is_none(is_none_) {}
    Point operator-()
    {
        return Point(-x, -y);
    }
    Point operator+(Point b)
    {
        return Point(x + b.x, y + b.y);
    }
    Point operator-(Point b)
    {
        return Point(x - b.x, y - b.y);
    }
    Point operator*(double scalar)
    {
        return Point(x * scalar, y * scalar);
    }
    Point operator/(double scalar)
    {
        return Point(x / scalar, y / scalar);
    }
    Point &operator+=(const Point other)
    {
        x += other.x;
        y += other.y;
        return *this;
    }
    Point &operator-=(const Point other)
    {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    Point &operator*=(const double other)
    {
        x *= other;
        y *= other;
        return *this;
    }
    Point &operator/=(const double other)
    {
        x /= other;
        y /= other;
        return *this;
    }
    bool operator==(Point other)
    {
        return (*this - other).mag() < EPSILON;
    }
    bool operator!=(Point other)
    {
        return !(*this == other);
    }
    double scalar(Point b)
    {
        return x * b.x + y * b.y;
    }

    double vector(Point b)
    {
        return x * b.y - y * b.x;
    }

    double mag2()
    {
        return x * x + y * y;
    }

    double mag()
    {
        return sqrt(mag2());
    }

    Point unity()
    {
        double len = mag();
        return len > 0.0f ? Point(x / len, y / len) : Point(0.0, 0.0);
    }

    double arg()
    {
        return atan2(y, x);
    }

    Point rotate(double ang)
    {
        double c = cos(ang), s = sin(ang);
        return Point(x * c - y * s, y * c + x * s);
    }
};

// iostream &operator<<(ostream &os, const Point &point)
// {
//     os << "x = " << point.x << ", y = " << point.y;
//     return os;
// }

Point GRAVEYARD_POS = Point(GRAVEYARD_POS_X, 0);

class Object
{
public:
    Point c;
    double r;
    Object() : c(Point(0, 0)), r(0) {}
    Object(Point c_, double r_) : c(c_), r(r_) {}
    bool operator==(Object other)
    {
        return c == other.c && r == other.r;
    }
    bool operator!=(Object other)
    {
        return !(*this == other);
    }
};

int sign(double a)
{
    if (a > 0)
        return 1;
    if (a < 0)
        return -1;
    return 0;
}

Point closest_point_on_line(Point point1, Point point2, Point point, char type = 'S')
{
    double line_len = (point1 - point2).mag();
    if (line_len == 0)
    {
        return point1;
    }
    Point line_dir = (point1 - point2).unity();
    Point point_vec = point - point1;
    double dot_product = point_vec.scalar(line_dir);
    if (dot_product <= 0 && type != 'L')
    {
        return point1;
    }
    if (dot_product >= line_len && type == 'S')
    {
        return point2;
    }
    return line_dir * dot_product + point1;
}

Point get_line_inretsesction(
    Point line1_start,
    Point line1_end,
    Point line2_start,
    Point line2_end,
    string types = "SS")
{
    double delta_x1 = line1_end.x - line1_start.x;
    double delta_y1 = line1_end.y - line1_start.y;
    double delta_x2 = line2_end.x - line2_start.x;
    double delta_y2 = line2_end.y - line2_start.y;
    double determinant = delta_y1 * delta_x2 - delta_y2 * delta_x1;
    if (determinant == 0)
        return Point(0, 0, true);
    double delta_x_start = line1_start.x - line2_start.x;
    double delta_y_start = line1_start.y - line2_start.y;
    double t1 = (delta_x_start * delta_y2 - delta_x2 * delta_y_start) / determinant;
    double t2 = (delta_x_start * delta_y1 - delta_x1 * delta_y_start) / determinant;
    double intersection_x = line1_start.x + t1 * delta_x1;
    double intersection_y = line1_start.y + t1 * delta_y1;
    Point p = Point(intersection_x, intersection_y);
    bool first_valid = false;
    bool second_valid = false;
    if ((types[0] == 'S' && 0 <= t1 && t1 <= 1) || (types[0] == 'R' && t1 >= 0) || types[0] == 'L')
        first_valid = true;
    if ((types[1] == 'S' && 0 <= t2 && t2 <= 1) || (types[1] == 'R' && t2 >= 0) || types[1] == 'L')
        second_valid = true;

    if (first_valid && second_valid)
        return p;
    return Point(0, 0, true);
}

double wind_down_angle(double angle)
{
    if (fabs(angle) > 2 * M_PI)
    {
        angle = fmod(angle, 2 * M_PI);
    }
    if (fabs(angle) > M_PI)
    {
        angle -= 2 * M_PI * sign(angle);
    }
    return angle;
}

double get_angle_between_points(Point a, Point b, Point c)
{
    return wind_down_angle((a - b).arg() - (c - b).arg());
}

void circles_inter(Point p0, Point p1, double r0, double r1, Point *out)
{
    double d = (p0 - p1).mag();
    double a = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
    double h = sqrtf(r0 * r0 - a * a);
    double x2 = p0.x + a * (p1.x - p0.x) / d;
    double y2 = p0.y + a * (p1.y - p0.y) / d;
    out[0].x = x2 + h * (p1.y - p0.y) / d;
    out[0].y = y2 - h * (p1.x - p0.x) / d;
    out[1].x = x2 - h * (p1.y - p0.y) / d;
    out[1].y = y2 + h * (p1.x - p0.x) / d;
}

int get_tangent_points(Point point0, Point point1, double r, Point *out)
{
    double d = (point1 - point0).mag();
    if (d < r)
    {
        return 0;
    }

    if (d == r)
    {
        out[0] = point1;
        return 1;
    }
    circles_inter(point0, Point((point0.x + point1.x) / 2, (point0.y + point1.y) / 2), r, d / 2, out);
    return 2;
}

Point nearest_point_on_poly(Point p, Point *poly, int ed_n)
{
    double min_ = -1, d;
    Point ans(0, 0), pnt(0, 0);
    for (int i = 0; i < ed_n; i++)
    {
        pnt = closest_point_on_line(poly[i], poly[i > 0 ? i - 1 : ed_n - 1], p);
        d = (pnt - p).mag();
        if (d < min_ || min_ < 0)
        {
            min_ = d;
            ans = pnt;
        }
    }
    return ans;
}

bool is_point_inside_poly(Point p, Point *points, int ed_n)
{
    double old_sign = sign((p - points[ed_n - 1]).vector(points[0] - points[ed_n - 1]));
    for (int i = 0; i < ed_n - 1; i++)
    {
        if (old_sign != sign((p - points[i]).vector(points[i + 1] - points[i])))
        {
            return false;
        }
    }
    return true;
}

int solve_one(double a, double b, complex<double> *out)
{
    if (a == 0)
    {
        return 0;
    }
    out[0] = -b / a;
    return 1;
}

int solve_two(double a, double b, double c, complex<double> *out)
{
    if (a == 0)
    {
        return solve_one(b, c, out);
        ;
    }
    static complex<double> D;
    D = complex<double>(b * b - 4 * a * c, 0.0);
    out[0] = (-b + sqrt(D)) / (2 * a);
    out[1] = (-b - sqrt(D)) / (2 * a);
    return 2;
}

int solve_three(double a, double b, double c, double d, complex<double> *out)
{
    if (a == 0)
    {
        return solve_two(b, c, d, out);
    }
    static double D0, D1, alpha;
    D0 = b * b - 3 * a * c;
    D1 = 2 * b * b * b - 9 * a * b * c + 27 * a * a * d;
    if (D0 == 0 && D1 == 0)
    {
        out[0] = -b / (3 * a);
        return 1;
    }
    static complex<double> m, high, low, e = complex<double>(-1 / 2.0, sqrt(3.0) / 2.0);
    m = sqrt(complex<double>(D1 * D1 - 4.0 * D0 * D0 * D0, 0));
    if (fabs(m) < EPSILON)
    {
        high = pow(D1, 1 / 3);
        low = D0 / high;
        out[0] = -(b + high * e + low / e) / (3.0 * a);
        alpha = fmod(arg(low) - arg(high), 2.0 * M_PI);
        if (fabs(alpha) > M_PI)
        {
            alpha -= 2.0 * M_PI * sign(alpha);
        }
        if (fabs(alpha) < M_PI / 4)
        {
            out[1] = -(b + high + low) / (3.0 * a);
        }
        else
        {
            out[1] = -(b + high * e * e + low / (e * e)) / (3.0 * a);
        }
        return 2;
    }
    if (fabs(m - D1) < EPSILON)
    {
        high = pow((D1 + m) / 2.0, 1.0 / 3.0);
    }
    else
    {
        high = pow((D1 - m) / 2.0, 1.0 / 3.0);
    }
    low = D0 / high;
    out[0] = -(b + high + low) / (3.0 * a);
    out[1] = -(b + high * e + low / e) / (3.0 * a);
    out[2] = -(b + high * e * e + low / (e * e)) / (3.0 * a);
    return 3;
}

int solve_four(double a, double b, double c, double d, double e, complex<double> *out)
{
    if (a == 0)
    {
        return solve_three(b, c, d, e, out);
    }
    b /= a;
    c /= a;
    d /= a;
    e /= a;
    static double p, q, r;
    p = (8.0 * c - 3.0 * b * b) / 8.0;
    q = (b * b * b - 4.0 * b * c + 8.0 * d) / 8.0;
    r = (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) / 256.0;
    static int i, n;
    n = solve_three(8.0, 8.0 * p, 2.0 * p * p - 8.0 * r, -q * q, out);
    static complex<double> m, k, h, l1, l2;
    m = complex<double>(0, 0);
    for (i = 0; i < n; i++)
    {
        if (fabs(out[i]) > EPSILON)
        {
            m = out[i];
            break;
        }
    }
    if (abs(m) == 0)
    {
        if (p == 0)
        {
            out[0] = -b / 4.0;
            return 1;
        }
        k = sqrt(-p / 2);
        out[0] = -b / 4.0 + k;
        out[1] = -b / 4.0 - k;
        return 2;
    }
    k = sqrt(2.0 * m);
    h = 2.0 * q / k;
    l1 = sqrt(-2.0 * p - 2.0 * m - h);
    l2 = sqrt(-2.0 * p - 2.0 * m + h);
    out[0] = -b / 4.0 + (k + l1) / 2.0;
    out[1] = -b / 4.0 + (k - l1) / 2.0;
    n = 2;
    if (fabs(2.0 * k - l1 - l2) > EPSILON && fabs(2.0 * k + l1 - l2) > EPSILON)
    {
        out[2] = -b / 4.0 + (-k + l2) / 2.0;
        n++;
    }
    if (fabs(2.0 * k + l1 + l2) > EPSILON && fabs(2.0 * k + l2 - l1) > EPSILON)
    {
        out[n] = -b / 4.0 + (-k - l2) / 2.0;
        n++;
    }
    return n;
}

Point closest_point_on_parabola(Point x, Point r0, Point v0, Point a, double t_min = -1e10, double t_max = 1e10)
{
    static double ak, bk, ck, dk, real_roots[3], best, answ, value;
    static int n_rls, n_rts, i;
    if (a.mag2() == 0 && v0.mag2() == 0)
    {
        return r0;
    }
    ak = a.mag2() / 2.0;
    bk = 3.0 / 2.0 * a.scalar(v0);
    ck = v0.mag2() + a.scalar(r0 - x);
    dk = v0.scalar(r0 - x);
    complex<double> roots[3];
    n_rts = solve_three(ak, bk, ck, dk, roots);
    n_rls = 0;
    for (i = 0; i < n_rts; i++)
    {
        if (abs(imag(roots[i])) < EPSILON)
        {
            real_roots[n_rls] = real(roots[i]);
            n_rls++;
        }
    }
    best = -1.0;
    for (i = 0; i < n_rls; i++)
    {
        if (real_roots[i] >= t_min && real_roots[i] <= t_max)
        {
            value = (r0 + v0 * real_roots[i] + a * real_roots[i] * real_roots[i] / 2.0 - x).mag2();
            if (best < 0 || value < best)
            {
                best = value;
                answ = real_roots[i];
            }
        }
    }
    value = (r0 + v0 * t_min + a * t_min * t_min / 2.0 - x).mag2();
    if (best < 0 || value < best)
    {
        best = value;
        answ = t_min;
    }
    value = (r0 + v0 * t_max + a * t_max * t_max / 2.0 - x).mag2();
    if (best < 0 || value < best)
    {
        best = value;
        answ = t_max;
    }
    return r0 + v0 * answ + a * answ * answ / 2.0;
}