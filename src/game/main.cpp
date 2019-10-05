#include "gameutils/tiled_map.h"

constexpr ivec2 screen_size(480, 270);
Interface::Window window("LD45", screen_size * 2, Interface::windowed, adjust_(Interface::WindowSettings{}, min_size = screen_size));
Graphics::DummyVertexArray dummy_vao = nullptr;

Audio::Context audio_context = nullptr;

Input::Mouse mouse;

Random random(std::time(0));

const Graphics::ShaderConfig shader_config = Graphics::ShaderConfig::Core();
Interface::ImGuiController gui_controller(Poly::derived<Interface::ImGuiController::GraphicsBackend_Modern>, adjust_(Interface::ImGuiController::Config{}, shader_header = shader_config.common_header));

Graphics::TextureAtlas texture_atlas(ivec2(2048), "assets/_images", "assets/atlas.png", "assets/atlas.refl");
Graphics::Texture texture_main = Graphics::Texture(nullptr).Wrap(Graphics::clamp).Interpolation(Graphics::nearest);

AdaptiveViewport adaptive_viewport(shader_config, screen_size);

using Renderer = Graphics::Renderers::Flat;
Renderer r(shader_config, 1000);

Graphics::Font font_main;
Graphics::FontFile font_file_main("assets/CatIV15.ttf", 15);

ReflectStruct(Sprites, (
    (Graphics::TextureAtlas::Region)(font_storage,tiles,player,sky,vignette),
))
Sprites sprites;

namespace Sounds
{
    #define SOUND_LIST(x) \
        x( jump       , 0.4 ) \
        x( land       , 0.4 ) \
        x( land_metal , 0.2 ) \

    #define X(name, random_pitch) \
        Audio::Buffer _buffer_##name(Audio::Sound(Audio::wav, Audio::mono, "assets/sounds/" #name ".wav")); \
        Audio::Source name(fvec2 pos, float vol = 1, float pitch = 1)                                       \
        {                                                                                                   \
            pitch = std::pow(2, std::log2(pitch) + float(-random_pitch <= random.real() <= random_pitch));  \
            return Audio::Source(_buffer_##name).temporary().volume(vol).pitch(pitch).pos(pos);             \
        }                                                                                                   \
        Audio::Source name(float vol = 1, float pitch = 1)                                                  \
        {                                                                                                   \
            return name(fvec2(0), vol, pitch).relative();                                                   \
        }                                                                                                   \

    SOUND_LIST(X)
    #undef X

    #undef SOUND_LIST
}

ReflectStruct(Controls, (
    (Input::Button)(left)(=Input::a),
    (Input::Button)(right)(=Input::d),
    (Input::Button)(jump)(=Input::w),
))
Controls controls;

constexpr int tile_size = 12;

constexpr int subpixel_bits = 6, subpixel_units = 1 << subpixel_bits;

inline namespace Level
{
    enum class Tile
    {
        air,
        stone,
        beam_v,
        beam_h,
        _count,
    };

    enum class TileStyle
    {
        invis,
        normal,
        smart,
    };

    struct TileInfo
    {
        TileStyle style = TileStyle::invis;
        int tex = 0;
        bool solid = 0;
    };

    const TileInfo &GetTileInfo(Tile tile)
    {
        static TileInfo info_array[]
        {
            /* air      */ adjust(TileInfo(), style = TileStyle::invis,           solid = 0 ),
            /* stone    */ adjust(TileInfo(), style = TileStyle::smart,  tex = 0, solid = 1 ),
            /* beam_v   */ adjust(TileInfo(), style = TileStyle::normal, tex = 0, solid = 1 ),
            /* beam_h   */ adjust(TileInfo(), style = TileStyle::normal, tex = 1, solid = 1 ),
        };

        static_assert(std::extent_v<decltype(info_array)> == int(Tile::_count));

        int tile_index = int(tile);
        if (tile_index < 0 || tile_index >= int(Tile::_count))
            Program::Error("Tile index is out of range.");

        return info_array[tile_index];
    }

    bool TileShouldMergeWith(Tile a, Tile b)
    {
        return a == b;
    }

    class Map
    {
      public:
        Tiled::TileLayer layer_mid;
        Tiled::PointLayer layer_points;

        Map() {}

        Map(MemoryFile file)
        {
            Json json(file.string(), 64);
            layer_mid = Tiled::LoadTileLayer(Tiled::FindLayer(json.GetView(), "mid"));

            for (ivec2 pos : vector_range(layer_mid.size()))
            {
                int tile = layer_mid.nonthrowing_at(pos);
                if (tile < 0 || tile >= int(Tile::_count))
                    Program::Error("In map `{}`: Invalid tile {} at [{},{}]."_format(file.name(), tile, pos.x, pos.y));
            }

            layer_points = Tiled::LoadPointLayer(Tiled::FindLayer(json.GetView(), "points"));
        }

        inline static std::vector<Map> all_maps;

        static void LoadAllMaps()
        {
            all_maps = {};
            int index = 0;

            while (1)
            {
                std::string file_name = "assets/maps/level{}.json"_format(index);

                bool found = true;
                Filesystem::GetObjectInfo(file_name, &found);
                if (!found)
                    break;

                all_maps.push_back(Map(file_name));

                index++;
            }

            if (index == 0)
                Program::Error("No maps found.");
        }

        ivec2 Size() const
        {
            return layer_mid.size();
        }

        static ivec2 PixelToTilePos(ivec2 pixel_pos)
        {
            return div_ex(pixel_pos, tile_size);
        }

        static Tile GetTile(ivec2 pos, const Tiled::TileLayer &layer)
        {
            return Tile(layer.clamped_at(pos));
        }
        Tile GetTileMid(ivec2 pos) const
        {
            return GetTile(pos, layer_mid);
        }

        static int GetRandomNumberForTile(ivec2 tile_pos, int size)
        {
            static const auto random_array = []{
                MultiArray<2, unsigned int> ret(ivec2(10,10));
                for (ivec2 pos : vector_range(ret.size()))
                    ret.nonthrowing_at(pos) = random.integer<unsigned int>();
                return ret;
            }();

            return random_array.nonthrowing_at(mod_ex(tile_pos, random_array.size())) % size;
        }

        static void RenderLayer(ivec2 pos, bool front_pass, const Tiled::TileLayer &layer)
        {
            ivec2 a = PixelToTilePos(pos - screen_size/2);
            ivec2 b = PixelToTilePos(pos + screen_size/2);

            for (ivec2 tile_pos : a <= vector_range <= b)
            {
                Tile tile = GetTile(tile_pos, layer);
                const TileInfo &info = GetTileInfo(tile);

                if (info.style == TileStyle::invis)
                    continue;

                ivec2 tile_pixel_pos = tile_pos * tile_size - pos;

                switch (info.style)
                {
                  case TileStyle::invis:
                    break;
                  case TileStyle::normal:
                    if (front_pass) break;
                    r << r.translate(tile_pixel_pos)
                        * r.TexturedQuad(sprites.tiles.region(ivec2(info.tex * tile_size, 0), ivec2(tile_size)));
                    break;
                  case TileStyle::smart:
                    if (!front_pass) break;
                    for (ivec2 offset : vector_range(ivec2(2)))
                    {
                        ivec2 pos_offset = offset*2-1;

                        int mask = 0b100 * TileShouldMergeWith(tile, GetTile(tile_pos with(x += pos_offset.x), layer))
                                 | 0b010 * TileShouldMergeWith(tile, GetTile(tile_pos + pos_offset, layer))
                                 | 0b001 * TileShouldMergeWith(tile, GetTile(tile_pos with(y += pos_offset.y), layer));

                        ivec2 tex_offset, tex_size = ivec2(1,1);

                        if (mask == 0b111) // Full block
                        {
                            int rand = GetRandomNumberForTile(tile_pos, 4);
                            rand = max(rand-1, 0);

                            tex_offset = ivec2(1 + rand, 4);
                        }
                        else if ((mask & 0b101) == 0b101) // Concave corner
                        {
                            tex_offset = ivec2(0,4);
                        }
                        else if (mask & 0b100) // Horizontal
                        {
                            tex_size = ivec2(1,2);
                            tex_offset = ivec2(2 + GetRandomNumberForTile(tile_pos, 2), 2);
                        }
                        else if (mask & 0b001) // Vertical
                        {
                            tex_size = ivec2(2,1);
                            tex_offset = ivec2(0, 2 + GetRandomNumberForTile(tile_pos, 2));
                        }
                        else // Corner
                        {
                            tex_size = ivec2(2,2);
                            tex_offset = ivec2(2 * GetRandomNumberForTile(tile_pos, 2), 0);
                        }

                        ivec2 tex_pixel_size = tex_size * tile_size/2;
                        ivec2 tex_pixel_pos = (ivec2(info.tex * 3, 1) + tex_offset) * tile_size + tex_pixel_size * offset;

                        r << r.translate(tile_pixel_pos + tile_size/2)
                            * r.TexturedQuad(sprites.tiles.region(tex_pixel_pos, tex_pixel_size)).CenterRel(1-offset);
                    }
                    break;
                }
            }
        }

        void RenderLayerMid(ivec2 pos, bool front_pass) const
        {
            RenderLayer(pos, front_pass, layer_mid);
        }
    };
}

namespace Draw
{
    struct TextData
    {
        Graphics::Text text;

        fvec2 pos;
        fvec3 color = fvec3(1);
        float alpha = 1;
        float beta = 1;
        ivec2 align = ivec2(0);
        int align_box_x = -2;

        std::optional<fmat3> matrix;
    };

    void Text(const TextData &data)
    {
        Graphics::Text::Stats stats = data.text.ComputeStats();

        ivec2 align_box(data.align_box_x >= -1 && data.align_box_x <= 1 ? data.align_box_x : data.align.x, data.align.y);

        fvec2 pos = data.pos;

        fvec2 offset = -stats.size * (1 + align_box) / 2;
        offset.x += stats.size.x * (1 + data.align.x) / 2; // Note that we don't change vertical position here.

        float line_start_offset_x = offset.x;

        for (size_t line_index = 0; line_index < data.text.lines.size(); line_index++)
        {
            const Graphics::Text::Line &line = data.text.lines[line_index];
            const Graphics::Text::Stats::Line &line_stats = stats.lines[line_index];

            offset.x = line_start_offset_x - line_stats.width * (1 + data.align.x) / 2;
            offset.y += line_stats.ascent;

            for (const Graphics::Text::Symbol &symbol : line.symbols)
            {
                fvec2 symbol_pos;

                if (!data.matrix)
                    symbol_pos = pos + offset + symbol.offset;
                else
                    symbol_pos = pos + (data.matrix.value() * (offset + symbol.offset).to_vec3(1)).to_vec2();

                r << r.translate(symbol_pos + (data.matrix ? data.matrix->z.to_vec2() : fvec2(0)))
                   * r.matrix(data.matrix ? data.matrix->to_mat2() : fmat2())
                   * r.TexturedQuad(symbol.texture_pos, symbol.size).Color(data.color, 1).Opacity(data.alpha, data.beta);

                offset.x += symbol.advance + symbol.kerning;
            }

            offset.y += line_stats.descent + line_stats.line_gap;
        }
    }
}

struct Hitbox
{
    ivec2 half_extent = ivec2(0);
    std::vector<ivec2> points;

    Hitbox() {}
    Hitbox(ivec2 size) : half_extent(size / 2)
    {
        for (ivec2 pos : vector_range((size - 2) / tile_size + 2))
            points.push_back(clamp_max(pos * tile_size, size-1) - half_extent);
    }

    template <typename F> void ForEachCollidingTile(const Tiled::TileLayer &layer, ivec2 pos, F &&func) const // `func` is `bool func(ivec2 tile_pos, Tile tile)`. If it returns `false`, the loop stops.
    {
        for (ivec2 point : points)
        {
            ivec2 tile_pos = Map::PixelToTilePos(pos + point);

            Tile tile = Map::GetTile(tile_pos, layer);
            bool should_continue = func(std::as_const(tile_pos), std::as_const(tile));

            if (!should_continue)
                break;
        }
    }

    bool IsSolidAt(const Tiled::TileLayer &layer, ivec2 pos) const
    {
        bool solid = 0;
        ForEachCollidingTile(layer, pos, [&](ivec2 /*tile_pos*/, Tile tile) -> bool
        {
            if (GetTileInfo(tile).solid)
            {
                solid = 1;
                return 0;
            }

            return 1;
        });
        return solid;
    }
};


class SubpixelPos
{
    ivec2 value_sub{};
    ivec2 value{};

    void UpdateValue()
    {
        value = value_sub >> subpixel_bits;
    }

  public:
    SubpixelPos() {}
    SubpixelPos(ivec2 new_value)
    {
        Set(new_value);
    }

    void Set(ivec2 new_value)
    {
        value = new_value;
        value_sub = new_value * subpixel_units + subpixel_units/2;
    }

    bool Offset(ivec2 subpixel_offset, std::function<bool(ivec2)> validate_pixel_pos = nullptr) // Returns `false` if the validation failed at some point.
    {
        if (!validate_pixel_pos)
        {
            value_sub += subpixel_offset;
            UpdateValue();
            return 1;
        }
        else
        {
            ivec2 offset_dir = sign(subpixel_offset);

            // Adjust position inside of the current pixel.
            for (int i = 0; i < 2; i++)
            {
                int &offset_dir_i = offset_dir[i];
                if (offset_dir_i == 0)
                    continue;

                int &offset_i = subpixel_offset[i];
                int &sub_i = value_sub[i];

                int dist_to_pixel_edge = (offset_dir_i > 0 ? subpixel_units - 1 - mod_ex(sub_i, subpixel_units) : mod_ex(sub_i, subpixel_units));
                clamp_var_max(dist_to_pixel_edge, abs(offset_i));

                sub_i += offset_dir_i * dist_to_pixel_edge;
                offset_i -= offset_dir_i * dist_to_pixel_edge;
            }

            DebugAssert("Fixed-point failure! (early)", div_ex(value_sub, subpixel_units) == value);

            // Adjust position on pixel scale.
            ivec2 pixel_offset = sign(subpixel_offset) * (abs(subpixel_offset) + subpixel_units - 1) / subpixel_units;

            while (1)
            {
                if ((pixel_offset - ivec2(0)).all()) // If can move diagonally
                {
                    ivec2 dir = sign(pixel_offset);
                    if (validate_pixel_pos(value + dir))
                    {
                        ivec2 sub_delta = dir * clamp_max(abs(subpixel_offset), subpixel_units);
                        value_sub += sub_delta;
                        subpixel_offset -= sub_delta;
                        pixel_offset -= dir;
                        UpdateValue();
                        continue;
                    }
                }

                bool can_move = 0;

                for (int i = 0; i < 2; i++)
                {
                    if (pixel_offset[i] == 0)
                        continue;

                    ivec2 dir(0);
                    dir[i] = sign(pixel_offset[i]);

                    if (validate_pixel_pos(value + dir))
                    {
                        ivec2 sub_delta = dir * clamp_max(abs(subpixel_offset), subpixel_units);
                        value_sub += sub_delta;
                        subpixel_offset -= sub_delta;
                        pixel_offset -= dir;
                        UpdateValue();
                        can_move = true;
                    }
                }

                if (!can_move)
                    break;
            }

            return pixel_offset == 0;
        }

        DebugAssert("Fixed-point failure!", div_ex(value_sub, subpixel_units) == value);

    }

    ivec2 Value() const
    {
        return value;
    }
    ivec2 SubpixelValue() const
    {
        return value_sub;
    }
};

struct Particle
{
    fvec2 pos{};
    fvec2 vel{};
    fvec2 acc{};
    fvec3 color = fvec3(1);
    float alpha = 1;
    float beta = 1;
    float size = 8;
    int current_time = 0;
    int mid_time = 30;
    int max_time = 60;
};

class ParticleController
{
    std::deque<Particle> particles;

  public:
    ParticleController() {}

    void Add(const Particle &particle)
    {
        particles.push_back(particle);
    }

    void Tick()
    {
        for (Particle &p : particles)
        {
            p.vel += p.acc;
            p.pos += p.vel;
            p.current_time++;
        }

        // Remove dead particles
        particles.erase(std::remove_if(particles.begin(), particles.end(), [](const Particle &p){return p.current_time >= p.max_time;}), particles.end());
    }

    void Render(ivec2 camera_pos) const
    {
        for (const Particle &p : particles)
        {
            float t = min(p.current_time / float(p.mid_time), 1 - (p.current_time - p.mid_time) / float(p.max_time - p.mid_time));
            float cur_size = smoothstep(t) * p.size;

            r << r.translate(p.pos - camera_pos) * r.UntexturedQuad(fvec2(cur_size)).Centered().Color(p.color).Opacity(p.alpha, p.beta);
        }
    }
};

struct Player
{
    inline static const Hitbox hitbox = Hitbox(ivec2(10, 19));

    SubpixelPos pos;
    fvec2 vel_subpixel{};
    int hc = 0;
    bool ground = 0;
    bool facing_left = 0;

    int stand_time = 0;
    int walk_time = 0;
};

struct World
{
    SubpixelPos camera_pos;
};

namespace States
{
    struct Base : Meta::with_virtual_destructor<Base>
    {
        virtual void Tick() = 0;
        virtual void Render() const = 0;
    };

    Poly::Storage<Base> current_state;

    struct Game : Base
    {
        Player p;
        World w;
        Map map;
        ParticleController par;

        int current_map_index = 0;

        Game()
        {
            SetMap(0);
        }

        void SetMap(int index)
        {
            if (index < 0 || index >= int(Map::all_maps.size()))
                Program::Error("Map index is out of range.");
            map = Map::all_maps[index];
            current_map_index = index;

            p.pos.Set(map.layer_points.GetSinglePoint("player"));
        }

        void Tick() override
        {
            static constexpr float gravity_subpixel = 9, jump_speed = 290, vel_cap_soft_up = 120, vel_cap_x = 150, vel_cap_y = 250,
                walk_speed = 90, walk_acc = 10, walk_acc_air = 8, walk_dec_fac = 0.75, walk_dec_fac_air = 0.9,
                particle_gravity = 0.01;

            // ImGui::InputFloat("gravity_subpixel", &gravity_subpixel, 1, 10);
            // ImGui::InputFloat("jump", &jump_speed, 1, 10);
            // ImGui::InputFloat("vel cap soft up", &vel_cap_soft_up, 1, 10);
            // ImGui::InputFloat("vel cap x", &vel_cap_x, 1, 10);
            // ImGui::InputFloat("vel cap y", &vel_cap_y, 1, 10);
            // ImGui::InputFloat("walk", &walk_speed, 1, 10);
            // ImGui::InputFloat("walk acc", &walk_acc, 1, 10);
            // ImGui::InputFloat("walk acc air", &walk_acc_air, 1, 10);
            // ImGui::InputFloat("walk dec fac", &walk_dec_fac, 0.01, 0.1);
            // ImGui::InputFloat("walk dec fac air", &walk_dec_fac_air, 0.01, 0.1);


            // ImGui::ShowDemoWindow();


            { // Player
                { // Check for ground
                    bool old_ground = p.ground;
                    p.ground = p.hitbox.IsSolidAt(map.layer_mid, p.pos.Value() with(y += 1));

                    // Landing effects
                    if (!old_ground && p.ground)
                    {
                        Tile tile = map.GetTileMid(Map::PixelToTilePos(p.pos.Value() with(y += p.hitbox.half_extent.y + 2)));
                        if (tile == Tile::beam_h || tile == Tile::beam_v)
                            Sounds::land_metal();
                        else
                            Sounds::land();

                        for (int i = 0; i < 8; i++)
                        {
                            par.Add(adjust(Particle{},
                                pos = p.pos.Value() with(y += p.hitbox.half_extent.y),
                                vel = fvec2::dir(f_pi / 12 <= random.real() <= f_pi / 3, 0.05 <= random.real() <= 0.4) with(x *= random.sign(), y *= -1),
                                acc = fvec2(0,particle_gravity),
                                current_time = 0,
                                mid_time = 5,
                                max_time = 40 <= random.integer() <= 60,
                                size = 1 <= random.real() <= 5,
                                color = fvec3(0.85 <= random.real() <= 1)
                            ));
                        }
                    }
                }

                // Walk
                p.hc = controls.right.down() - controls.left.down();
                if (p.hc)
                {
                    clamp_var(p.vel_subpixel.x += p.hc * (p.ground ? walk_acc : walk_acc_air), -walk_speed, walk_speed);
                    p.facing_left = p.hc < 0;
                    p.walk_time++;
                    p.stand_time = 0;
                }
                else
                {
                    p.vel_subpixel.x *= p.ground ? walk_dec_fac : walk_dec_fac_air;
                    p.walk_time = 0;
                    p.stand_time++;
                }

                // Jump
                if (controls.jump.pressed())
                {
                    p.vel_subpixel.y = -jump_speed;
                    Sounds::jump();

                    for (int i = 0; i < 8; i++)
                    {
                        par.Add(adjust(Particle{},
                            pos = p.pos.Value() with(y += p.hitbox.half_extent.y),
                            vel = fvec2::dir(f_pi / 12 <= random.real() <= f_pi / 3, 0.05 <= random.real() <= 0.4) with(x *= random.sign(), y *= -1),
                            acc = fvec2(0,particle_gravity),
                            current_time = 0,
                            mid_time = 5,
                            max_time = 40 <= random.integer() <= 60,
                            size = 1 <= random.real() <= 5,
                            color = fvec3(0.85 <= random.real() <= 1)
                        ));
                    }
                }

                if (controls.jump.released() && p.vel_subpixel.y)
                    clamp_var_min(p.vel_subpixel.y, -vel_cap_soft_up);


                // Apply gravity
                p.vel_subpixel.y += gravity_subpixel;

                // Clamp velocity
                clamp_var(p.vel_subpixel, -ivec2(vel_cap_x, vel_cap_y), ivec2(vel_cap_x, vel_cap_y));

                fvec2 clamped_vel_subpixel = p.vel_subpixel;
                clamp_var_min(clamped_vel_subpixel.y, -vel_cap_soft_up);

                // Move
                p.pos.Offset(clamped_vel_subpixel, [&](ivec2 new_pos)
                {
                    return !p.hitbox.IsSolidAt(map.layer_mid, new_pos);
                });

                // Set velocity to null on impact
                for (int i = 0; i < 2; i++)
                {
                    auto &vel_i = p.vel_subpixel[i];
                    ivec2 dir{};
                    dir[i] = sign(vel_i);

                    if (p.hitbox.IsSolidAt(map.layer_mid, p.pos.Value() + dir))
                        vel_i = 0;
                }
            }

            { // Camera
                w.camera_pos.Set(p.pos.Value());

                Audio::Listener::Position(w.camera_pos.Value().to_vec3(-3 * screen_size.x/2));
                Audio::Listener::Orientation(fvec3(0,0,1), fvec3(0,-1,0));
            }

            { // Particles
                par.Tick();
            }
        }

        void Render() const override
        {
            constexpr int stand_anim_ticks_per_frame = 15, stand_anim_frames = 5, walk_anim_ticks_per_frame = 5, walk_anim_frames = 7;

            Graphics::Clear();

            // Background
            // r << r.UntexturedQuad(screen_size).Color(fvec3(0x7e, 0xa9, 0xff) / 255).Centered();
            r << r.TexturedQuad(sprites.sky).Centered();


            // Draw::Text(adjust(Draw::TextData{}, text = Graphics::Text(font_main, "Hello world!\nHello!"), pos = mouse.pos(), color = fvec3(1,0.5,0)));

            // Map (back)
            map.RenderLayerMid(w.camera_pos.Value(), false);

            { // Player
                int anim_state = 0;
                int anim_frame = 0;

                if (p.ground)
                {
                    if (p.hc)
                    {
                        // Walking
                        anim_state = 1;
                        anim_frame = (p.walk_time / walk_anim_ticks_per_frame) % walk_anim_frames;
                    }
                    else
                    {
                        // Idle
                        anim_state = 0;
                        anim_frame = (p.stand_time / stand_anim_ticks_per_frame) % stand_anim_frames;
                    }
                }
                else
                {
                    // Jumping
                    anim_state = 2;
                    if (p.vel_subpixel.y < -100)
                        anim_frame = 0;
                    else if (p.vel_subpixel.y < -70)
                        anim_frame = 1;
                    else if (p.vel_subpixel.y < -40)
                        anim_frame = 2;
                    else if (p.vel_subpixel.y < 80)
                        anim_frame = 3;
                    else if (p.vel_subpixel.y < 170)
                        anim_frame = 4;
                    else
                        anim_frame = 5;
                }

                constexpr ivec2 player_sprite_size(24);
                r << r.translate(p.pos.Value() - w.camera_pos.Value() + ivec2(0,-1)) * r.flip_x(p.facing_left)
                   * r.TexturedQuad(sprites.player.region(player_sprite_size * ivec2(anim_frame, anim_state), player_sprite_size)).Centered();
            }

            // Particles
            par.Render(w.camera_pos.Value());

            // Map (front)
            map.RenderLayerMid(w.camera_pos.Value(), true);

            // Vignette
            r << r.TexturedQuad(sprites.vignette).Centered();


            r.Flush();
        }
    };
}

int ENTRY_POINT(int, char **)
{
    { // Initialize
        { // Renderer
            Graphics::Blending::Enable();
            r.SetBlendingMode();
            Graphics::SetClearColor(fvec3(0));

            { // Load sprites from the atlas
                using refl_t = Refl::Interface<Sprites>;
                refl_t refl(sprites);

                refl.for_each_field([&](auto index)
                {
                    constexpr auto i = index.value;

                    std::string image_name = "{}.png"_format(refl.field_name(i));
                    refl.field_value<i>() = texture_atlas.Get(image_name);
                });
            }

            { // Load fonts
                Unicode::CharSet ranges;
                ranges.Add(Unicode::Ranges::Basic_Latin);
                ranges.Add(Unicode::Ranges::Latin_1_Supplement);

                Graphics::MakeFontAtlas(texture_atlas.GetImage(), sprites.font_storage.pos, sprites.font_storage.size, {
                    Graphics::FontAtlasEntry(font_main, font_file_main, ranges, Graphics::FontFile::light),
                });
            }

            { // Upload final texture
                texture_main.SetData(texture_atlas.GetImage());
                r.SetTexture(texture_main);
            }
        }

        { // Gui
            ImGui::StyleColorsDark();

            // Load various small fonts
            auto monochrome_font_flags = ImGuiFreeType::Monochrome | ImGuiFreeType::MonoHinting;

            gui_controller.LoadFont("assets/Monokat_6x12.ttf", 12.0f, adjust(ImFontConfig{}, RasterizerFlags = monochrome_font_flags));
            gui_controller.LoadDefaultFont();
            gui_controller.RenderFontsWithFreetype();
        }

        { // Audio
            Audio::Source::DefaultRefDistance(4 * screen_size.x / 2);
            Audio::Source::DefaultMaxDistance(4 * screen_size.x / 2);
            Audio::Source::DefaultRolloffFactor(1);
        }

        Map::LoadAllMaps();
    }

    auto Resize = [&]
    {
        adaptive_viewport.Update();
        r.SetMatrix(adaptive_viewport.GetDetails().MatrixCentered());
        mouse.SetMatrix(adaptive_viewport.GetDetails().MouseMatrixCentered());
    };
    Resize();

    States::current_state = Poly::derived<States::Game>;

    Metronome metronome(60);
    Clock::DeltaTimer delta_timer;

    while (1)
    {
        uint64_t delta = delta_timer();
        while (metronome.Tick(delta))
        {
            // window.ProcessEvents();
            window.ProcessEvents({gui_controller.EventHook()});

            if (window.Resized())
            {
                Resize();
                Graphics::Viewport(window.Size());
            }
            if (window.ExitRequested())
                Program::Exit();

            gui_controller.PreTick();
            States::current_state->Tick();
            audio_context.Tick();
        }

        gui_controller.PreRender();
        adaptive_viewport.BeginFrame();
        States::current_state->Render();
        adaptive_viewport.FinishFrame();
        Graphics::CheckErrors();
        gui_controller.PostRender();

        window.SwapBuffers();
    }
}
