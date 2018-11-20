#pragma once

#include <memory>

#include "program/errors.h"
#include "utils/mat.h"

namespace Graphics
{
    struct ShaderConfig;
    class TexUnit;
    class Texture;
}

class Render
{
    struct Data;
    std::unique_ptr<Data> data;

    void *GetRenderQueuePtr();

  public:
    Render(decltype(nullptr));
    Render(int queue_size, const Graphics::ShaderConfig &config);

    ~Render();

    explicit operator bool() const;

    void BindShader() const;

    void Finish();

    void SetTextureUnit(const Graphics::TexUnit &unit);
    void SetTextureUnit(Graphics::TexUnit &&) = delete;

    void SetTextureSize(ivec2 size);

    void SetTexture(const Graphics::Texture &tex);
    void SetTexture(Graphics::Texture &&) = delete;

    void SetMatrix(const fmat4 &m);

    void SetColorMatrix(const fmat4 &m);

    class Quad_t
    {
        friend class Render;

        using ref = Quad_t &&;

        void *queue = 0; // Actually the type should be `Graphics::RenderQueue<Attribs, 3> *`, but we don't include "graphics/render_queue.h" for better compilation times.

        struct Data
        {
            fvec2 pos, size; // The constructor sets these.

            bool has_texture = 0;
            fvec2 tex_pos = fvec2(0), tex_size = fvec2(0);

            bool has_center = 0;
            fvec2 center = fvec2(0);
            bool center_pos_tex = 0;

            bool has_matrix = 0;
            fmat3 matrix = fmat3();

            bool has_color = 0;
            fvec3 colors[4] {};

            bool has_tex_color_fac = 0;
            float tex_color_factors[4] = {1,1,1,1};

            float alpha[4] = {1,1,1,1};
            float beta[4] = {1,1,1,1};

            bool abs_pos = 0;
            bool abs_tex_pos = 0;

            bool flip_x = 0, flip_y = 0;
        };
        Data data;

        Quad_t(void *queue, fvec2 pos, fvec2 size) : queue(queue)
        {
            data.pos = pos;
            data.size = size;
        }
      public:
        Quad_t(Quad_t &&other) noexcept : queue(std::exchange(other.queue, {})), data(std::move(other.data)) {}
        Quad_t &operator=(Quad_t other)
        {
            std::swap(queue, other.queue);
            std::swap(data, other.data);
            return *this;
        }

        ~Quad_t();

        ref tex(fvec2 pos, fvec2 size)
        {
            DebugAssert("2D poly renderer: Quad_t texture specified twice.", !data.has_texture);
            data.has_texture = 1;

            data.tex_pos = pos;
            data.tex_size = size;
            return (ref)*this;
        }
        ref tex(fvec2 pos)
        {
            tex(pos, data.size);
            return (ref)*this;
        }
        ref center(fvec2 c)
        {
            DebugAssert("2D poly renderer: Quad_t center specified twice.", !data.has_center);
            data.has_center = 1;

            data.center = c;
            data.center_pos_tex = 1;
            return (ref)*this;
        }
        ref pixel_center(fvec2 c) // Same as `center()`, but the coordinates are always measured in pixels instead of texels even if a texture is specified.
        {
            DebugAssert("2D poly renderer: Quad_t center specified twice.", !data.has_center);
            data.has_center = 1;

            data.center = c;
            data.center_pos_tex = 0;
            return (ref)*this;
        }
        ref center()
        {
            pixel_center(data.size / 2);
            return (ref)*this;
        }
        ref matrix(fmat3 m) // This can be called multiple times, resulting in multiplying matrices in the order they were passed.
        {
            if (data.has_matrix)
            {
                data.matrix = data.matrix * m;
            }
            else
            {
                data.has_matrix = 1;
                data.matrix = m;
            }
            return (ref)*this;
        }
        ref matrix(fmat2 m)
        {
            matrix(m.to_mat3());
            return (ref)*this;
        }
        ref rotate(float a) // Uses `matrix()`.
        {
            matrix(fmat3::rotate(a));
            return (ref)*this;
        }
        ref translate(fvec2 v) // Uses a matrix.
        {
            matrix(fmat3::translate(v));
            return (ref)*this;
        }
        ref scale(fvec2 s) // Uses a matrix.
        {
            matrix(fmat3::scale(s));
            return (ref)*this;
        }
        ref scale(float s) // Uses a matrix.
        {
            scale(fvec2(s));
            return (ref)*this;
        }
        ref color(fvec3 c)
        {
            DebugAssert("2D poly renderer: Quad_t color specified twice.", !data.has_color);
            data.has_color = 1;

            for (auto &it : data.colors)
                it = c;
            return (ref)*this;
        }
        ref color(fvec3 a, fvec3 b, fvec3 c, fvec3 d)
        {
            DebugAssert("2D poly renderer: Quad_t color specified twice.", !data.has_color);
            data.has_color = 1;

            data.colors[0] = a;
            data.colors[1] = b;
            data.colors[2] = c;
            data.colors[3] = d;
            return (ref)*this;
        }
        ref mix(float x) // 0 - fill with color, 1 - use texture
        {
            DebugAssert("2D poly renderer: Quad_t texture/color factor specified twice.", !data.has_tex_color_fac);
            data.has_tex_color_fac = 1;

            for (auto &it : data.tex_color_factors)
                it = x;
            return (ref)*this;
        }
        ref mix(float a, float b, float c, float d)
        {
            DebugAssert("2D poly renderer: Quad_t texture/color factor specified twice.", !data.has_tex_color_fac);
            data.has_tex_color_fac = 1;

            data.tex_color_factors[0] = a;
            data.tex_color_factors[1] = b;
            data.tex_color_factors[2] = c;
            data.tex_color_factors[3] = d;
            return (ref)*this;
        }
        ref alpha(float a)
        {
            for (auto &it : data.alpha)
                it = a;
            return (ref)*this;
        }
        ref alpha(float a, float b, float c, float d)
        {
            data.alpha[0] = a;
            data.alpha[1] = b;
            data.alpha[2] = c;
            data.alpha[3] = d;
            return (ref)*this;
        }
        ref beta(float a) // 1 - normal blending, 0 - additive blending
        {
            for (auto &it : data.beta)
                it = a;
            return (ref)*this;
        }
        ref beta(float a, float b, float c, float d)
        {
            data.beta[0] = a;
            data.beta[1] = b;
            data.beta[2] = c;
            data.beta[3] = d;
            return (ref)*this;
        }
        ref absolute(bool x = 1) // Interpret size as a position of the second corner
        {
            data.abs_pos = x;
            return (ref)*this;
        }
        ref absolute_tex(bool x = 1) // Interpret texture size as a position of the second corner
        {
            data.abs_tex_pos = x;
            return (ref)*this;
        }
        ref flip_x(bool f = 1) // Flips texture horizontally if it was specified. Updates the center accordingly if it was specified.
        {
            data.flip_x = f;
            return (ref)*this;
        }
        ref flip_y(bool f = 1) // Flips texture vertically if it was specified. Updates the center accordingly if it was specified.
        {
            data.flip_y = f;
            return (ref)*this;
        }
    };

    class Triangle_t
    {
        friend class Render;

        using ref = Triangle_t &&;

        void *queue = 0; // Actually the type should be `Graphics::RenderQueue<Attribs, 3> *`, but we don't include "graphics/render_queue.h" for better compilation times.

        struct Data
        {
            fvec2 pos[3]; // The constructor sets these.

            bool has_texture = 0;
            fvec2 tex_pos[3] = {};

            bool has_matrix = 0;
            fmat3 matrix = fmat3();

            bool has_color = 0;
            fvec3 colors[3] {};

            bool has_tex_color_fac = 0;
            float tex_color_factors[3] = {1,1,1};

            float alpha[3] = {1,1,1};
            float beta[3] = {1,1,1};
        };
        Data data;

        Triangle_t(void *queue, fvec2 a, fvec2 b, fvec2 c) : queue(queue)
        {
            data.pos[0] = a;
            data.pos[1] = b;
            data.pos[2] = c;
        }
      public:
        Triangle_t(Triangle_t &&other) noexcept : queue(std::exchange(other.queue, {})), data(std::move(other.data)) {}
        Triangle_t &operator=(Triangle_t other)
        {
            std::swap(queue, other.queue);
            std::swap(data, other.data);
            return *this;
        }

        ~Triangle_t();

        ref tex(fvec2 a, fvec2 b, fvec2 c)
        {
            DebugAssert("2D poly renderer: Triangle_t texture specified twice.", !data.has_texture);
            data.has_texture = 1;

            data.tex_pos[0] = a;
            data.tex_pos[1] = b;
            data.tex_pos[2] = c;
            return (ref)*this;
        }
        ref matrix(fmat3 m) // This can be called multiple times, resulting in multiplying matrices in the order they were passed.
        {
            if (data.has_matrix)
            {
                data.matrix = data.matrix * m;
            }
            else
            {
                data.has_matrix = 1;
                data.matrix = m;
            }
            return (ref)*this;
        }
        ref matrix(fmat2 m)
        {
            matrix(m.to_mat3());
            return (ref)*this;
        }
        ref rotate(float a) // Uses `matrix()`.
        {
            matrix(fmat3::rotate(a));
            return (ref)*this;
        }
        ref translate(fvec2 v) // Uses a matrix.
        {
            matrix(fmat3::translate(v));
            return (ref)*this;
        }
        ref scale(fvec2 s) // Uses a matrix.
        {
            matrix(fmat3::scale(s));
            return (ref)*this;
        }
        ref scale(float s) // Uses a matrix.
        {
            scale(fvec2(s));
            return (ref)*this;
        }
        ref color(fvec3 x)
        {
            return color(x,x,x);
        }
        ref color(fvec3 a, fvec3 b, fvec3 c)
        {
            DebugAssert("2D poly renderer: Triangle_t color specified twice.", !data.has_color);
            data.has_color = 1;

            data.colors[0] = a;
            data.colors[1] = b;
            data.colors[2] = c;
            return (ref)*this;
        }
        ref mix(float x) // 0 - fill with color, 1 - use texture
        {
            return mix(x,x,x);
        }
        ref mix(float a, float b, float c)
        {
            DebugAssert("2D poly renderer: Triangle_t texture/color factor specified twice.", !data.has_tex_color_fac);
            data.has_tex_color_fac = 1;

            data.tex_color_factors[0] = a;
            data.tex_color_factors[1] = b;
            data.tex_color_factors[2] = c;
            return (ref)*this;
        }
        ref alpha(float x)
        {
            return alpha(x,x,x);
        }
        ref alpha(float a, float b, float c)
        {
            data.alpha[0] = a;
            data.alpha[1] = b;
            data.alpha[2] = c;
            return (ref)*this;
        }
        ref beta(float x) // 1 - normal blending, 0 - additive blending
        {
            return beta(x, x, x);
        }
        ref beta(float a, float b, float c)
        {
            data.beta[0] = a;
            data.beta[1] = b;
            data.beta[2] = c;
            return (ref)*this;
        }
    };

    Quad_t fquad(fvec2 pos, fvec2 size)
    {
        return Quad_t(GetRenderQueuePtr(), pos, size);
    }
    Quad_t iquad(ivec2 pos, ivec2 size)
    {
        return Quad_t(GetRenderQueuePtr(), pos, size);
    }
    Quad_t iquad(fvec2 pos, fvec2 size) = delete;

    Triangle_t ftriangle(fvec2 a, fvec2 b, fvec2 c)
    {
        return Triangle_t(GetRenderQueuePtr(), a, b, c);
    }
    Triangle_t itriangle(ivec2 a, ivec2 b, ivec2 c)
    {
        return Triangle_t(GetRenderQueuePtr(), a, b, c);
    }
    Triangle_t itriangle(fvec2 a, fvec2 b, fvec2 c) = delete;
};
