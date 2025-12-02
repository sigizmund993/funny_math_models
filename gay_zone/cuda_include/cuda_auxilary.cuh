#pragma once
// optimized geometric helpers for CUDA/CPU
// float precision, __host__ __device__, forceinline, minimal branches
#include <cmath>
#include <cfloat>
#include <stdint.h>
#include "const.cuh"

struct __align__(8) Point {
    float x;
    float y;
    bool is_none;

    __host__ __device__ __forceinline__ Point() noexcept : x(0.0f), y(0.0f), is_none(false) {}
    __host__ __device__ __forceinline__ Point(float x_, float y_, bool none_ = false) noexcept
        : x(x_), y(y_), is_none(none_) {}
    // basic arithmetic
    __host__ __device__ __forceinline__ Point operator-() const noexcept { return Point(-x, -y); }
    __host__ __device__ __forceinline__ Point operator+(const Point &b) const noexcept { return Point(x + b.x, y + b.y); }
    __host__ __device__ __forceinline__ Point operator-(const Point &b) const noexcept { return Point(x - b.x, y - b.y); }
    __host__ __device__ __forceinline__ Point operator*(float s) const noexcept { return Point(x * s, y * s); }
    __host__ __device__ __forceinline__ Point operator/(float s) const noexcept { float inv = 1.0f / s; return Point(x * inv, y * inv); }

    __host__ __device__ __forceinline__ Point &operator+=(const Point &o) noexcept { x += o.x; y += o.y; return *this; }
    __host__ __device__ __forceinline__ Point &operator-=(const Point &o) noexcept { x -= o.x; y -= o.y; return *this; }
    __host__ __device__ __forceinline__ Point &operator*=(float s) noexcept { x *= s; y *= s; return *this; }
    __host__ __device__ __forceinline__ Point &operator/=(float s) noexcept { float inv = 1.0f / s; x *= inv; y *= inv; return *this; }

    __host__ __device__ __forceinline__ bool operator==(const Point &o) const noexcept {
        float dx = x - o.x; float dy = y - o.y; return (dx*dx + dy*dy) < (EPSILON * EPSILON);
    }
    __host__ __device__ __forceinline__ bool operator!=(const Point &o) const noexcept { return !(*this == o); }

    __host__ __device__ __forceinline__ float scalar(const Point &b) const noexcept { return x * b.x + y * b.y; }
    __host__ __device__ __forceinline__ float vector(const Point &b) const noexcept { return x * b.y - y * b.x; }
    __host__ __device__ __forceinline__ float mag2() const noexcept { return x * x + y * y; }
    __host__ __device__ __forceinline__ float mag() const noexcept { return sqrtf(mag2()); }

    __host__ __device__ __forceinline__ Point unity() const noexcept {
        float len = mag();
        return (len > 0.0f) ? Point(x / len, y / len) : Point(0.0f, 0.0f);
    }

    __host__ __device__ __forceinline__ float arg() const noexcept { return atan2f(y, x); }

    __host__ __device__ __forceinline__ Point rotate(float ang) const noexcept {
        float c = cosf(ang), s = sinf(ang);
        return Point(x * c - y * s, y * c + x * s);
    }
};


__host__ __device__ __forceinline__ Point get_graveyard_pos() noexcept {
#ifdef GRAVEYARD_POS_X
    return Point((float)GRAVEYARD_POS_X, 0.0f);
#else
    return Point(0.0f, 0.0f);
#endif
}

struct Object {
    Point c;
    float r;
    __host__ __device__ __forceinline__ Object() noexcept : c(Point()), r(0.0f) {}
    __host__ __device__ __forceinline__ Object(const Point &cc, float rr) noexcept : c(cc), r(rr) {}
    __host__ __device__ __forceinline__ bool operator==(const Object &o) const noexcept { return c == o.c && fabsf(r - o.r) < EPSILON; }
    __host__ __device__ __forceinline__ bool operator!=(const Object &o) const noexcept { return !(*this == o); }
};

// branchless-ish sign for float
__host__ __device__ __forceinline__ int signf(float a) noexcept {
    return (a > 0.0f) - (a < 0.0f);
}

__host__ __device__ __forceinline__ Point closest_point_on_line(const Point &p1, const Point &p2, const Point &pt, char type = 'S') noexcept {
    // returns closest point on segment (default), ray or line depending on type
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float line_len2 = dx*dx + dy*dy;
    if (line_len2 <= EPSILON) return p1;
    float inv_len = rsqrtf(line_len2); // faster reciprocal sqrt if available
    float line_len = 1.0f / inv_len;   // = sqrt(line_len2)
    // line_dir = (p2 - p1) / line_len
    float lx = dx * inv_len;
    float ly = dy * inv_len;
    float vx = pt.x - p1.x;
    float vy = pt.y - p1.y;
    float dot = vx * lx + vy * ly;
    if (type != 'L') {
        if (dot <= 0.0f) return p1;
        if (type == 'S' && dot >= line_len) return p2;
    }
    return Point(p1.x + lx * dot, p1.y + ly * dot);
}

__host__ __device__ __forceinline__ Point get_line_intersection(
    const Point &l1s, const Point &l1e,
    const Point &l2s, const Point &l2e,
    char type1 = 'S', char type2 = 'S') noexcept
{
    float dx1 = l1e.x - l1s.x; float dy1 = l1e.y - l1s.y;
    float dx2 = l2e.x - l2s.x; float dy2 = l2e.y - l2s.y;
    float det = dy1 * dx2 - dy2 * dx1;
    if (fabsf(det) <= EPSILON) return Point(0.0f, 0.0f, true);
    float dxs = l1s.x - l2s.x; float dys = l1s.y - l2s.y;
    float t1 = (dxs * dy2 - dx2 * dys) / det;
    float t2 = (dxs * dy1 - dx1 * dys) / det;
    bool ok1 = (type1 == 'L') || (type1 == 'R' && t1 >= -EPSILON) || (type1 == 'S' && t1 >= -EPSILON && t1 <= 1.0f + EPSILON);
    bool ok2 = (type2 == 'L') || (type2 == 'R' && t2 >= -EPSILON) || (type2 == 'S' && t2 >= -EPSILON && t2 <= 1.0f + EPSILON);
    if (ok1 && ok2) {
        return Point(l1s.x + t1 * dx1, l1s.y + t1 * dy1);
    }
    return Point(0.0f, 0.0f, true);
}

__host__ __device__ __forceinline__ float wind_down_anglef(float angle) noexcept {
    // reduce to (-pi, pi]
    // use fmodf safely
    if (fabsf(angle) > 2.0f * M_PI) angle = fmodf(angle, 2.0f * M_PI);
    if (angle > M_PI) angle -= 2.0f * M_PI;
    else if (angle <= -M_PI) angle += 2.0f * M_PI;
    return angle;
}

__host__ __device__ __forceinline__ float get_angle_between_points(const Point &a, const Point &b, const Point &c) noexcept {
    // angle at b between ba and bc
    float ang = (a - b).arg() - (c - b).arg();
    return wind_down_anglef(ang);
}

__host__ __device__ __forceinline__ void circles_inter(const Point &p0, const Point &p1, float r0, float r1, Point out[2]) noexcept {
    // robust intersection of two circles, writes into out[], may write undefined if no intersection
    float dx = p1.x - p0.x; float dy = p1.y - p0.y;
    float d2 = dx*dx + dy*dy;
    float d = sqrtf(d2);
    if (d <= EPSILON) { out[0] = Point(0.0f,0.0f,true); out[1] = out[0]; return; }
    float a = (r0*r0 - r1*r1 + d2) / (2.0f * d);
    float h2 = r0*r0 - a*a;
    if (h2 < -EPSILON) { out[0] = Point(0.0f,0.0f,true); out[1] = out[0]; return; }
    if (h2 < 0.0f) h2 = 0.0f;
    float x2 = p0.x + a * (dx) / d;
    float y2 = p0.y + a * (dy) / d;
    float rx = -dy * (sqrtf(h2) / d);
    float ry = dx  * (sqrtf(h2) / d);
    out[0] = Point(x2 + rx, y2 + ry);
    out[1] = Point(x2 - rx, y2 - ry);
}

__host__ __device__ __forceinline__ int get_tangent_points(const Point &p0, const Point &p1, float r, Point *out) noexcept {
    // tangent points from circle centered at p0 radius r to point p1 (returns number of tangent points)
    float dx = p1.x - p0.x; float dy = p1.y - p0.y;
    float d2 = dx*dx + dy*dy;
    float d = sqrtf(d2);
    if (d < r - EPSILON) return 0;
    if (fabsf(d - r) <= EPSILON) { out[0] = p1; return 1; }
    // construct circle intersection with helper
    Point mid((p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f);
    Point tmp[2];
    circles_inter(p0, mid, r, d * 0.5f, tmp);
    // if circles_inter failed, fallback to geometric construction
    if (tmp[0].is_none) {
        // fallback: compute angles
        float ang = atan2f(dy, dx);
        float alpha = acosf(r / d);
        out[0] = Point(p0.x + r * cosf(ang + alpha), p0.y + r * sinf(ang + alpha));
        out[1] = Point(p0.x + r * cosf(ang - alpha), p0.y + r * sinf(ang - alpha));
        return 2;
    }
    out[0] = tmp[0];
    out[1] = tmp[1];
    return 2;
}

__host__ __device__ __forceinline__ Point nearest_point_on_poly(const Point &p, const Point *poly, int ed_n) noexcept {
    float min_d2 = FLT_MAX;
    Point ans(0.0f, 0.0f);
    for (int i = 0; i < ed_n; ++i) {
        const Point &a = poly[i];
        const Point &b = poly[(i + ed_n - 1) % ed_n];
        Point cand = closest_point_on_line(a, b, p, 'S');
        float d2 = (cand.x - p.x)*(cand.x - p.x) + (cand.y - p.y)*(cand.y - p.y);
        if (d2 < min_d2) { min_d2 = d2; ans = cand; }
    }
    return ans;
}

__host__ __device__ __forceinline__ bool is_point_inside_poly(const Point &p, const Point *pts, int ed_n) noexcept {
    // using winding sign test (assumes convex polygon in original code). Keep same semantics:
    if (ed_n < 3) return false;
    int old_sign = signf((p - pts[ed_n - 1]).vector(pts[0] - pts[ed_n - 1]));
    for (int i = 0; i < ed_n - 1; ++i) {
        int s = signf((p - pts[i]).vector(pts[i + 1] - pts[i]));
        if (s != old_sign) return false;
    }
    return true;
}



