#pragma once

#include <SFML/Graphics.hpp>
#include "auxilary.h"
#include "field.h"

using namespace sf;

class Drawer
{
public:
    RenderWindow window;
    double scale = 0.1;
    Drawer(int w, int h) : window(VideoMode(w, h), "venom") {}
    void drawCircle(Point pos, double radius, Color color)
    {
        pos.y *= -1;
        pos += Point(FIELD_DX / 2.0 + FIELD_MARGIN, FIELD_DY / 2.0 + FIELD_MARGIN);
        pos *= scale;
        radius *= scale;
        CircleShape circle(radius);
        circle.setFillColor(color);
        circle.setPosition(pos.x - radius, pos.y - radius);
        window.draw(circle);
    }
    void drawRectangle(Point pos, float width, float height, Color color) {
        pos.y *=-1;
        pos += Point(FIELD_DX+FIELD_MARGIN,FIELD_DY+FIELD_MARGIN);
        pos *=scale;
        width *=scale;
        height *=scale;
        
        RectangleShape rect(Vector2f(width, height));
        rect.setFillColor(color);
        rect.setPosition(pos.x, pos.y);
        window.draw(rect);
    }
    void drawLine(Point p1, Point p2, float thickness, Color color) {
        p1.y *=-1;
        p1 += Point(FIELD_DX+FIELD_MARGIN,FIELD_DY+FIELD_MARGIN);
        p1 *=scale;
        p2.y *=-1;
        p2 += Point(FIELD_DX+FIELD_MARGIN,FIELD_DY+FIELD_MARGIN);
        p2 *=scale;
        thickness *= scale;
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float length = std::sqrt(dx * dx + dy * dy);
        float angle = std::atan2(dy, dx) * 180 / 3.14159265f;

        RectangleShape line(Vector2f(length, thickness));
        line.setFillColor(color);
        line.setPosition(p1.x, p1.y);
        line.setRotation(angle);

        window.draw(line);
    }
    void update()
    {
        drawLine(Point(-FIELD_DX,FIELD_DY),Point(FIELD_DX,FIELD_DY),20,Color(200,200,200));
        drawLine(Point(FIELD_DX,FIELD_DY),Point(FIELD_DX,-FIELD_DY),20,Color(200,200,200));
        drawLine(Point(FIELD_DX,-FIELD_DY),Point(-FIELD_DX,-FIELD_DY),20,Color(200,200,200));
        drawLine(Point(-FIELD_DX,-FIELD_DY),Point(-FIELD_DX,FIELD_DY),20,Color(200,200,200));
        window.display();        
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed)
                window.close();
        }
        window.clear(FIELD_COLOR);
    }
};
