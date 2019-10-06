#include "gameutils/tiled_map.h"

constexpr ivec2 screen_size(480, 270);
Interface::Window window("WADJ — LD45", screen_size * 2, Interface::windowed, adjust_(Interface::WindowSettings{}, min_size = screen_size));
Graphics::DummyVertexArray dummy_vao = nullptr;

Audio::Context audio_context = nullptr;

Input::Mouse mouse;

Random random(std::time(0));

const Graphics::ShaderConfig shader_config = Graphics::ShaderConfig::Core();
Interface::ImGuiController gui_controller(Poly::derived<Interface::ImGuiController::GraphicsBackend_Modern>,
    adjust_(Interface::ImGuiController::Config{}, shader_header = shader_config.common_header, store_state_in_file = ""));

Graphics::TextureAtlas texture_atlas(ivec2(2048), "assets/_images", "assets/atlas.png", "assets/atlas.refl");
Graphics::Texture texture_main = Graphics::Texture(nullptr).Wrap(Graphics::clamp).Interpolation(Graphics::nearest);

AdaptiveViewport adaptive_viewport(shader_config, screen_size);

using Renderer = Graphics::Renderers::Flat;
Renderer r(shader_config, 1000);

Graphics::Font font_main;
Graphics::FontFile font_file_main("assets/CatIV15.ttf", 15);

ReflectStruct(Sprites, (
    (Graphics::TextureAtlas::Region)(font_storage,tiles,player,sky,vignette,cursor,frame,click_me,letters,small_letters,key_enabled,bullet,final_box),
))
Sprites sprites;

namespace Sounds
{
    #define SOUND_LIST(x) \
        x( jump       , 0.4 ) \
        x( land       , 0.4 ) \
        x( jump_metal , 0.2 ) \
        x( land_metal , 0.2 ) \
        x( death      , 0.4 ) \
        x( power_on   , 0.3 ) \
        x( power_off  , 0.3 ) \
        x( powerup    , 0.2 ) \
        x( wrong_key  , 0.3 ) \
        x( checkpoint , 0.3 ) \
        x( laser      , 0.3 ) \
        x( laser_hit  , 0.3 ) \
        x( box_hit    , 0.3 ) \

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

    Audio::Buffer theme_buffer(Audio::Sound(Audio::ogg, "assets/sounds/theme.ogg"));
    Audio::Source theme_src;
}

ReflectStruct(Controls, (
    (Input::Button)(left)(=Input::a),
    (Input::Button)(right)(=Input::d),
    (Input::Button)(jump)(=Input::w),
    (Input::Button)(shoot)(=Input::j),
    (Input::Button)(reset)(=Input::escape),
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
        spike,
        bg_stone,
        bg_beam_v,
        bg_beam_h,
        wire,
        button,
        opt_block,
        barrier,
        slot_w,
        slot_a,
        slot_d,
        slot_j,
        target,
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
        bool front = 0;
        int tex = 0;
        bool solid = 0;
        bool kills = 0;
        bool is_mid = 0, is_bg = 0, is_wire = 0;
        bool elec = 0;
        int tex_elec = 0;
        bool solid_elec = 0;
    };

    const TileInfo &GetTileInfo(Tile tile)
    {
        static TileInfo info_array[]
        {
            /* air       */ adjust(TileInfo(), style = TileStyle::invis                                                                                      ),
            /* stone     */ adjust(TileInfo(), style = TileStyle::smart, front = 1, tex = 0,  solid = 1, is_mid = 1                                          ),
            /* beam_v    */ adjust(TileInfo(), style = TileStyle::normal,           tex = 0,  solid = 1, is_mid = 1                                          ),
            /* beam_h    */ adjust(TileInfo(), style = TileStyle::normal,           tex = 1,  solid = 1, is_mid = 1                                          ),
            /* spike     */ adjust(TileInfo(), style = TileStyle::normal,           tex = 2,  kills = 1, is_mid = 1                                          ),
            /* bg_stone  */ adjust(TileInfo(), style = TileStyle::smart, front = 1, tex = 1,             is_bg = 1                                           ),
            /* bg_beam_v */ adjust(TileInfo(), style = TileStyle::normal,           tex = 3,             is_bg = 1                                           ),
            /* bg_beam_h */ adjust(TileInfo(), style = TileStyle::normal,           tex = 4,             is_bg = 1                                           ),
            /* wire      */ adjust(TileInfo(), style = TileStyle::smart, front = 1, tex = 2,             is_wire = 1, elec = 1, tex_elec = 3                 ),
            /* button    */ adjust(TileInfo(), style = TileStyle::normal,           tex = 5,             is_wire = 1, elec = 1, tex_elec = 6                 ),
            /* opt_block */ adjust(TileInfo(), style = TileStyle::smart,/* !front */tex = 4,             is_mid = 1,  elec = 1, tex_elec = 5,  solid_elec = 1),
            /* barrier   */ adjust(TileInfo(), style = TileStyle::invis,                      kills = 1, is_mid = 1                                          ),
            /* slot_w    */ adjust(TileInfo(), style = TileStyle::normal,           tex = 11,            is_wire = 1, elec = 1, tex_elec = 7                 ),
            /* slot_a    */ adjust(TileInfo(), style = TileStyle::normal,           tex = 12,            is_wire = 1, elec = 1, tex_elec = 8                 ),
            /* slot_d    */ adjust(TileInfo(), style = TileStyle::normal,           tex = 13,            is_wire = 1, elec = 1, tex_elec = 9                 ),
            /* slot_j    */ adjust(TileInfo(), style = TileStyle::normal,           tex = 14,            is_wire = 1, elec = 1, tex_elec = 10                ),
            /* target    */ adjust(TileInfo(), style = TileStyle::normal,           tex = 15, solid = 1, is_mid = 1,  elec = 1, tex_elec = 16, solid_elec = 1),
        };

        static_assert(std::extent_v<decltype(info_array)> == int(Tile::_count));

        int tile_index = int(tile);
        if (tile_index < 0 || tile_index >= int(Tile::_count))
            Program::Error("Tile index is out of range.");

        return info_array[tile_index];
    }

    bool TileIsSlot(Tile t)
    {
        return t == Tile::slot_w || t == Tile::slot_a || t == Tile::slot_d || t == Tile::slot_j;
    }

    class Map
    {
      public:
        Tiled::TileLayer layer_mid, layer_back, layer_wire;
        Tiled::PointLayer layer_points;

        void ForEachTileLayer(std::function<void(Tiled::TileLayer &)> func)
        {
            func(layer_mid);
            func(layer_back);
            func(layer_wire);
        }
        void ForEachTileLayer(std::function<void(const Tiled::TileLayer &)> func) const
        {
            func(layer_mid);
            func(layer_back);
            func(layer_wire);
        }

        struct WireState
        {
            static constexpr int max_dist_to_source = std::numeric_limits<int>::max();

            int dist_to_source = max_dist_to_source;
            int prev_dist_to_source = max_dist_to_source;

            bool IsEnabled() const
            {
                return dist_to_source != max_dist_to_source;
            }
        };

        MultiArray<2, WireState> wire_state;
        std::unordered_set<ivec2> wire_elec_sources;

        ivec2 click_me_sign{};

        bool TilesShouldMerge(ivec2 a_pos, Tile a, ivec2 b_pos, Tile b) const
        {
            (void)a_pos;

            auto IsWireLike = [](Tile t)
            {
                return t == Tile::wire || t == Tile::button || t == Tile::opt_block || TileIsSlot(t) || t == Tile::target;
            };

            if (IsWireLike(a) && a != Tile::opt_block)
            {
                bool found = 0;
                ForEachTileLayer([&](const Tiled::TileLayer &la)
                {
                    Tile other_tile = GetTile(b_pos, la);
                    if (IsWireLike(other_tile))
                        found = 1;
                });
                if (found)
                    return 1;
            }

            return a == b;
        }

        Map() {}

        Map(MemoryFile file)
        {
            Json json(file.string(), 64);
            layer_mid = Tiled::LoadTileLayer(Tiled::FindLayer(json.GetView(), "mid"));
            layer_back = Tiled::LoadTileLayer(Tiled::FindLayer(json.GetView(), "back"));
            layer_wire = Tiled::LoadTileLayer(Tiled::FindLayer(json.GetView(), "wire"));

            if (layer_mid.size() != layer_back.size() || layer_mid.size() != layer_wire.size())
                Program::Error("In map `{}`: Layers have different size."_format(file.name()));

            wire_state = decltype(wire_state)(layer_mid.size());

            for (auto *la : {&layer_mid, &layer_back, &layer_wire})
            for (ivec2 pos : vector_range(layer_mid.size()))
            {
                int tile = la->nonthrowing_at(pos);

                if (tile < 0 || tile >= int(Tile::_count))
                    Program::Error("In map `{}`: Invalid tile {} at [{},{}]."_format(file.name(), tile, pos.x, pos.y));

                if (tile != 0)
                {
                    const auto &info = GetTileInfo(Tile(tile));
                    bool at_proper_layer = la == &layer_mid ? info.is_mid :
                                           la == &layer_back ? info.is_bg :
                                           la == &layer_wire ? info.is_wire : true;
                    if (!at_proper_layer)
                        Program::Error("In map `{}`: Tile at [{},{}] can't be at this layer."_format(file.name(), pos.x, pos.y));

                    if (TileIsSlot(Tile(tile)))
                    {
                        bool ok = wire_elec_sources.insert(pos).second;
                        if (!ok)
                            Program::Error("In map `{}`: Duplicate power source at [{},{}]."_format(file.name(), pos.x, pos.y));
                    }
                }
            }

            layer_points = Tiled::LoadPointLayer(Tiled::FindLayer(json.GetView(), "points"));

            layer_points.ForEachPointNamed("power", [&](fvec2 pos)
            {
                ivec2 tile_pos = iround(trunc(pos / tile_size));
                bool ok = wire_elec_sources.insert(tile_pos).second;
                if (!ok)
                    Program::Error("In map `{}`: Duplicate power source at [{},{}]."_format(file.name(), tile_pos.x, tile_pos.y));
            });

            click_me_sign = layer_points.GetSinglePoint("click_me");
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
        Tile GetTileBack(ivec2 pos) const
        {
            return GetTile(pos, layer_back);
        }
        Tile GetTileWire(ivec2 pos) const
        {
            return GetTile(pos, layer_wire);
        }
        WireState &WireStateAt(ivec2 pos)
        {
            return wire_state.clamped_at(pos);
        }
        const WireState &WireStateAt(ivec2 pos) const
        {
            return wire_state.clamped_at(pos);
        }

        bool HaveSolidTileAt(ivec2 pos) const
        {
            Tile tile = GetTileMid(pos);
            const auto &info = GetTileInfo(tile);

            return info.elec && WireStateAt(pos).IsEnabled() ? info.solid_elec : info.solid;
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

        void RenderLayer(ivec2 pos, bool front_pass, const Tiled::TileLayer &layer, const decltype(wire_state) &wire_state) const
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

                int tile_tex = info.tex;
                if (info.elec && wire_state.clamped_at(tile_pos).IsEnabled())
                    tile_tex = info.tex_elec;

                switch (info.style)
                {
                  case TileStyle::invis:
                    break;
                  case TileStyle::normal:
                    if (front_pass != info.front) break;
                    r << r.translate(tile_pixel_pos)
                        * r.TexturedQuad(sprites.tiles.region(ivec2(tile_tex * tile_size, 0), ivec2(tile_size)));
                    break;
                  case TileStyle::smart:
                    if (front_pass != info.front) break;
                    for (ivec2 offset : vector_range(ivec2(2)))
                    {
                        ivec2 pos_offset = offset*2-1;

                        int mask = 0b100 * TilesShouldMerge(tile_pos, tile, tile_pos with(x += pos_offset.x),
                                                                    GetTile(tile_pos with(x += pos_offset.x), layer))
                                 | 0b010 * TilesShouldMerge(tile_pos, tile, tile_pos + pos_offset,
                                                                    GetTile(tile_pos + pos_offset, layer))
                                 | 0b001 * TilesShouldMerge(tile_pos, tile, tile_pos with(y += pos_offset.y),
                                                                    GetTile(tile_pos with(y += pos_offset.y), layer));

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
                        ivec2 tex_pixel_pos = (ivec2(tile_tex * 4, 1) + tex_offset) * tile_size + tex_pixel_size * offset;

                        r << r.translate(tile_pixel_pos + tile_size/2)
                            * r.TexturedQuad(sprites.tiles.region(tex_pixel_pos, tex_pixel_size)).CenterRel(1-offset);
                    }
                    break;
                }
            }
        }

        void RenderLayerMid(ivec2 pos, bool front_pass) const
        {
            RenderLayer(pos, front_pass, layer_mid, wire_state);
        }
        void RenderLayerBack(ivec2 pos, bool front_pass) const
        {
            RenderLayer(pos, front_pass, layer_back, wire_state);
        }
        void RenderLayerWire(ivec2 pos, bool front_pass) const
        {
            RenderLayer(pos, front_pass, layer_wire, wire_state);
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

    bool IsSolidAt(const Map &map, ivec2 pos) const
    {
        bool solid = 0;
        ForEachCollidingTile(map.layer_mid, pos, [&](ivec2 tile_pos, Tile) -> bool
        {
            if (map.HaveSolidTileAt(tile_pos))
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
    std::deque<Particle> particles_back, particles_front;

  public:
    ParticleController() {}

    void AddBack(const Particle &particle)
    {
        particles_back.push_back(particle);
    }
    void AddFront(const Particle &particle)
    {
        particles_front.push_back(particle);
    }

    void Reset()
    {
        particles_back = {};
        particles_front = {};
    }

    void Tick()
    {
        for (auto *list_ptr : {&particles_back, &particles_front})
        {
            auto &list = *list_ptr;

            for (Particle &p : list)
            {
                p.vel += p.acc;
                p.pos += p.vel;
                p.current_time++;
            }

            // Remove dead particles
            list.erase(std::remove_if(list.begin(), list.end(), [](const Particle &p){return p.current_time >= p.max_time;}), list.end());
        }
    }

    void Render(ivec2 camera_pos, bool front) const
    {
        for (const Particle &p : front ? particles_front : particles_back)
        {
            float t = min(p.current_time / float(p.mid_time), 1 - (p.current_time - p.mid_time) / float(p.max_time - p.mid_time));
            float cur_size = smoothstep(t) * p.size;

            r << r.translate(p.pos - camera_pos) * r.UntexturedQuad(fvec2(cur_size)).Centered().Color(p.color).Opacity(p.alpha, p.beta);
        }
    }
};

struct Player
{
    inline static const Hitbox hitbox = Hitbox(ivec2(10, 19)), damage_hitbox = Hitbox(ivec2(10, 10));

    SubpixelPos pos;
    fvec2 vel_subpixel{};
    int hc = 0;
    bool ground = 1;
    bool facing_left = 0;

    int stand_time = 0;
    int walk_time = 0;

    bool alive = 1;
    int death_timer = 0;

    int fire_cooldown = 0;

    std::set<char> enabled_controls;
};

struct LetterPowerup
{
    inline static const Hitbox hitbox = Hitbox(ivec2(14, 14));

    SubpixelPos pos;
    fvec2 vel_subpixel{};
    bool ground = 1;

    char ch = ' ';

    int TexIndex() const
    {
        constexpr const char *symbols = "wadj";
        const char *ptr = std::strchr(symbols, ch);
        return ptr ? ptr - symbols : -1;
    }
};

struct Bullet
{
    SubpixelPos pos;
    fvec2 vel{};
    int age = 0;
};

struct World
{
    SubpixelPos camera_pos;
    fvec2 camera_vel{};

    float fade_in = 1, death_fade_out = 0;
    bool need_wire_update = 1;

    int cursor_idle_timer = 1000000;
    ivec2 cursor_pos{};

    bool clicked_at_least_once = 0;
    float click_me_sign_alpha = 1;

    bool pressed_at_least_once = 0;
    float key_enabled_sign_alpha = 1;

    std::vector<LetterPowerup> letter_powerups;

    std::vector<ivec2> checkpoints;
    std::vector<Bullet> bullets;

    ivec2 final_box_pos{};
    static constexpr int final_box_max_hp = 5;
    int final_box_hp = final_box_max_hp;
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

        static Game saved_game;

        Game()
        {
        }

        void Init()
        {
            map = Map("assets/map.json");

            p.pos.Set(iround(map.layer_points.GetSinglePoint("player") with(y -= 4)));
            w.camera_pos = p.pos.Value();
            w.final_box_pos = iround(map.layer_points.GetSinglePoint("final_box"));

            map.layer_points.ForEachPointWithNamePrefix("letter:", [&](std::string name, fvec2 pos)
            {
                if (name.size() != 8)
                    Program::Error("Invalid letter object at [{},{}]"_format(Map::PixelToTilePos(pos).x, Map::PixelToTilePos(pos).y));
                auto &new_letter = w.letter_powerups.emplace_back();
                new_letter.pos.Set(pos);
                new_letter.ch = name[7];
                if (!std::strchr("wadj", new_letter.ch))
                    Program::Error("Invalid letter object at [{},{}]"_format(Map::PixelToTilePos(pos).x, Map::PixelToTilePos(pos).y));
            });

            map.layer_points.ForEachPointNamed("checkpoint", [&](fvec2 pos)
            {
                w.checkpoints.push_back(iround(pos));
            });

            saved_game = *this;
        }

        void Tick() override
        {
            constexpr
            static float gravity_subpixel = 9, jump_speed = 290, vel_cap_soft_up = 120, vel_up_cap_factor = 0.1, vel_cap_x = 150, vel_cap_y = 250,
                walk_speed = 90, walk_acc = 10, walk_acc_air = 8, walk_dec_fac = 0.75, walk_dec_fac_air = 0.9,
                particle_gravity = 0.01,
                camera_y_offset = 16, camera_ref_dist = 10, camera_acc_pow = 1.5, camera_drag_fac = 0.8, camera_mass = 34,
                fade_step = 0.02, death_fade_step = 0.035,
                click_me_sign_alpha_step = 0.08,
                bullet_speed = 4;

            // for (char ch : std::string("wadj"))
            // {
            //     bool state = p.enabled_controls.count(ch);
            //     bool old_state = state;
            //     ImGui::Checkbox("[{}]"_format(ch).c_str(), &state);
            //     if (state != old_state)
            //     {
            //         if (state)
            //             p.enabled_controls.insert(ch);
            //         else
            //             p.enabled_controls.erase(ch);
            //     }
            // }

            // ImGui::InputFloat("gravity_subpixel", &gravity_subpixel, 1, 10);
            // ImGui::InputFloat("jump", &jump_speed, 1, 10);
            // ImGui::InputFloat("vel cap soft up", &vel_cap_soft_up, 1, 10);
            // ImGui::InputFloat("vel cap soft up (factor)", &vel_up_cap_factor, 0.01, 0.1);
            // ImGui::InputFloat("vel cap x", &vel_cap_x, 1, 10);
            // ImGui::InputFloat("vel cap y", &vel_cap_y, 1, 10);
            // ImGui::InputFloat("walk", &walk_speed, 1, 10);
            // ImGui::InputFloat("walk acc", &walk_acc, 1, 10);
            // ImGui::InputFloat("walk acc air", &walk_acc_air, 1, 10);
            // ImGui::InputFloat("walk dec fac", &walk_dec_fac, 0.01, 0.1);
            // ImGui::InputFloat("walk dec fac air", &walk_dec_fac_air, 0.01, 0.1);
            // ImGui::InputFloat("camera y offset", &camera_y_offset, 1, 10);
            // ImGui::InputFloat("camera ref dist", &camera_ref_dist, 1, 10);
            // ImGui::InputFloat("camera acc pow", &camera_acc_pow, 0.01, 0.1);
            // ImGui::InputFloat("camera drag fac", &camera_drag_fac, 0.01, 0.1);
            // ImGui::InputFloat("camera mass", &camera_mass, 1, 10);

            // ImGui::ShowDemoWindow();

            { // Wires
                if (w.need_wire_update)
                {
                    w.need_wire_update = 0;

                    struct DelayedUpdate
                    {
                        ivec2 pos{};
                        int skip_if_dist_leq = 0;
                    };
                    std::vector<DelayedUpdate> delayed_updates;

                    auto StartWireUpdate = [&](ivec2 pos, bool is_source, int skip_if_dist_leq)
                    {
                        { // Check if we should update this wire
                            if (!map.layer_wire.pos_in_range(pos))
                                return; // This wire is out of range.

                            bool have_wire_at_this_tile = 0;
                            map.ForEachTileLayer([&](Tiled::TileLayer &la)
                            {
                                if (GetTileInfo(Map::GetTile(pos, la)).elec)
                                    have_wire_at_this_tile = 1;
                            });
                            if(!have_wire_at_this_tile)
                                return; // No wire at this tile.
                        }

                        Map::WireState &state = map.WireStateAt(pos);
                        if (state.dist_to_source <= skip_if_dist_leq)
                            return;

                        int old_dist = state.dist_to_source;

                        // Update signal strength
                        if (is_source)
                        {
                            state.dist_to_source = 0;
                        }
                        else
                        {
                            int min_adj_dist = state.max_dist_to_source;

                            for (int i = 0; i < 4; i++)
                            {
                                ivec2 offset = ivec2(1,0).rot90(i);
                                if (!map.layer_wire.pos_in_range(pos + offset))
                                    continue; // This wire is out of range, skip it.

                                const Map::WireState &other_state = map.WireStateAt(pos + offset);

                                if (other_state.dist_to_source < min_adj_dist)
                                    min_adj_dist = other_state.dist_to_source;
                            }

                            if (min_adj_dist != state.max_dist_to_source)
                                state.dist_to_source = min_adj_dist + 1;
                            else
                                state.dist_to_source = state.max_dist_to_source;
                        }

                        if (state.dist_to_source != state.prev_dist_to_source)
                            w.need_wire_update = 1;

                        if (state.dist_to_source != old_dist)
                        {
                            for (int i = 0; i < 4; i++)
                                delayed_updates.push_back(adjust(DelayedUpdate{}, pos = pos + ivec2(1,0).rot90(i), skip_if_dist_leq = state.dist_to_source));
                        }
                    };

                    // Reset wires.
                    for (ivec2 pos : vector_range(map.Size()))
                    {
                        auto &state = map.WireStateAt(pos);
                        state.prev_dist_to_source = state.dist_to_source;
                        state.dist_to_source = Map::WireState::max_dist_to_source;
                    }

                    // Recalculate wires.
                    for (ivec2 pos : map.wire_elec_sources)
                        StartWireUpdate(pos, true, -1);

                    while (delayed_updates.size() > 0)
                    {
                        std::vector<DelayedUpdate> copied_updates = std::move(delayed_updates);
                        delayed_updates.clear();
                        for (const auto &upd : copied_updates)
                            StartWireUpdate(upd.pos, false, upd.skip_if_dist_leq);
                    }
                }
            }

            { // Player
                // Check for ground
                if (p.alive)
                {
                    bool old_ground = p.ground;
                    p.ground = p.hitbox.IsSolidAt(map, p.pos.Value() with(y += 1));

                    // Landing effects
                    if (!old_ground && p.ground)
                    {
                        Tile tile = map.GetTileMid(Map::PixelToTilePos(p.pos.Value() with(y += p.hitbox.half_extent.y + 2)));
                        if (tile == Tile::beam_h || tile == Tile::beam_v || tile == Tile::opt_block)
                            Sounds::land_metal();
                        else
                            Sounds::land();

                        for (int i = 0; i < 8; i++)
                        {
                            par.AddBack(adjust(Particle{},
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
                p.hc = (controls.right.down() * p.enabled_controls.count('d') - controls.left.down() * p.enabled_controls.count('a')) * p.alive;
                if (p.hc)
                {
                    clamp_var(p.vel_subpixel.x += p.hc * (p.ground ? walk_acc : walk_acc_air), -walk_speed, walk_speed);
                    p.facing_left = p.hc < 0;
                    p.walk_time++;
                    p.stand_time = 0;
                    w.pressed_at_least_once = 1;
                }
                else
                {
                    p.vel_subpixel.x *= p.ground ? walk_dec_fac : walk_dec_fac_air;
                    p.walk_time = 0;
                    p.stand_time++;
                }

                // Jump
                if (p.alive && p.enabled_controls.count('w'))
                {
                    if (p.ground && controls.jump.pressed())
                    {
                        p.vel_subpixel.y = -jump_speed;

                        Tile tile = map.GetTileMid(Map::PixelToTilePos(p.pos.Value() with(y += p.hitbox.half_extent.y + 2)));
                        if (tile == Tile::beam_h || tile == Tile::beam_v || tile == Tile::opt_block)
                            Sounds::jump_metal();
                        else
                            Sounds::jump();

                        for (int i = 0; i < 8; i++)
                        {
                            par.AddBack(adjust(Particle{},
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

                    if (controls.jump.released() && p.enabled_controls.count('w') && p.vel_subpixel.y)
                        clamp_var_min(p.vel_subpixel.y, -vel_cap_soft_up);
                }

                // Shoot
                {
                    if (p.alive && p.enabled_controls.count('j') && p.fire_cooldown == 0 && controls.shoot.pressed())
                    {
                        Sounds::laser();
                        p.fire_cooldown = 20;

                        w.bullets.push_back(adjust(Bullet{}, pos.Set(p.pos.Value() + ivec2((p.facing_left ? -1 : 1) * 4, -5)), vel = fvec2((p.facing_left ? -1 : 1) * bullet_speed, 0)));
                    }

                    if (p.fire_cooldown > 0)
                        p.fire_cooldown--;
                }


                // Apply gravity
                p.vel_subpixel.y += gravity_subpixel;

                // Clamp velocity
                clamp_var(p.vel_subpixel, -ivec2(vel_cap_x, vel_cap_y), ivec2(vel_cap_x, vel_cap_y));

                fvec2 clamped_vel_subpixel = p.vel_subpixel;
                if (clamped_vel_subpixel.y < -vel_cap_soft_up)
                    clamped_vel_subpixel.y = (clamped_vel_subpixel.y + vel_cap_soft_up) * vel_up_cap_factor - vel_cap_soft_up;

                // Move
                p.pos.Offset(clamped_vel_subpixel, [&](ivec2 new_pos)
                {
                    return !p.hitbox.IsSolidAt(map, new_pos);
                });

                // Push out of walls
                if (p.hitbox.IsSolidAt(map, p.pos.Value()))
                {
                    // Try pushing player out of the wall first

                    bool in_wall = 1;
                    if (in_wall) // Vertical
                    {
                        for (int d = 1; d <= 8; d++)
                        {
                            for (int s = -1; s <= 1; s += 2)
                            {
                                ivec2 new_pos = p.pos.Value() with(y += s * d);
                                if (!p.hitbox.IsSolidAt(map, new_pos))
                                {
                                    p.pos.Set(new_pos);
                                    in_wall = 0;
                                    break;
                                }
                            }

                            if (!in_wall)
                                break;
                        }
                    }
                    if (in_wall) // Horizontal
                    {
                        for (int d = 1; d <= 6; d++)
                        {
                            for (int s = -1; s <= 1; s += 2)
                            {
                                ivec2 new_pos = p.pos.Value() with(x += s * d);
                                if (!p.hitbox.IsSolidAt(map, new_pos))
                                {
                                    p.pos.Set(new_pos);
                                    in_wall = 0;
                                    break;
                                }
                            }

                            if (!in_wall)
                                break;
                        }
                    }
                }

                // Set velocity to null on impact
                for (int i = 0; i < 2; i++)
                {
                    auto &vel_i = p.vel_subpixel[i];
                    ivec2 dir{};
                    dir[i] = sign(vel_i);

                    if (p.hitbox.IsSolidAt(map, p.pos.Value() + dir))
                        vel_i = 0;
                }

                { // Check for wrong controls
                    if (p.enabled_controls.count('w') == 0 && controls.jump.pressed())
                        Sounds::wrong_key();
                    if (p.enabled_controls.count('a') == 0 && controls.left.pressed())
                        Sounds::wrong_key();
                    if (p.enabled_controls.count('d') == 0 && controls.right.pressed())
                        Sounds::wrong_key();
                    if (p.enabled_controls.count('j') == 0 && controls.shoot.pressed())
                        Sounds::wrong_key();
                }

                { // Check for letter slots
                    ivec2 tile_pos = Map::PixelToTilePos(p.pos.Value());
                    Tile tile = map.GetTileWire(tile_pos);
                    if (TileIsSlot(tile))
                    {
                        auto source_it = map.wire_elec_sources.find(tile_pos);

                        if (source_it != map.wire_elec_sources.end())
                        {
                            char slot_ch =
                                tile == Tile::slot_w ? 'w' :
                                tile == Tile::slot_a ? 'a' :
                                tile == Tile::slot_d ? 'd' :
                                                       'j';

                            bool have_this_ch = p.enabled_controls.count(slot_ch);

                            if (have_this_ch)
                            {
                                p.enabled_controls.erase(slot_ch);
                                map.wire_elec_sources.erase(source_it);
                                w.need_wire_update = 1;

                                ivec2 pixel_pos = tile_pos * tile_size + tile_size/2;
                                Sounds::power_off(pixel_pos);

                                for (int i = 0; i < 14; i++)
                                {
                                    constexpr fvec3 color_a = fvec3(0x58, 0x2e, 0x13) / 255, color_b = fvec3(0xfb, 0x96, 0x32) / 255;

                                    par.AddFront(adjust(Particle{},
                                        pos = pixel_pos + fvec2(-4 <= random.real() <= 4, -4 <= random.real() <= 4),
                                        vel = fvec2::dir(random.angle(), 0.05 <= random.real() <= 0.7),
                                        acc = fvec2(0,particle_gravity*2),
                                        current_time = 0,
                                        mid_time = 5,
                                        max_time = 20 <= random.integer() <= 40,
                                        size = 1 <= random.real() <= 3.5,
                                        color = random.boolean() ? color_b : color_a,
                                        beta = 0.75
                                    ));
                                }
                            }
                        }
                    }
                }

                // Check damage sources
                if (p.alive)
                {
                    // Spikes
                    bool touches_damage_source = 0;
                    p.damage_hitbox.ForEachCollidingTile(map.layer_mid, p.pos.Value(), [&](ivec2 /*tile_pos*/, Tile tile) -> bool
                    {
                        if (GetTileInfo(tile).kills)
                        {
                            touches_damage_source = 1;
                            return 0;
                        }
                        return 1;
                    });

                    // Map bounds
                    if (!touches_damage_source && ((p.pos.Value() < 0).any() || (p.pos.Value() >= map.Size() * tile_size).any()))
                        touches_damage_source = 1;

                    // Hotkey
                    if (!touches_damage_source && controls.reset.pressed())
                        touches_damage_source = 1;

                    // Suffocation
                    if (!touches_damage_source && p.hitbox.IsSolidAt(map, p.pos.Value()))
                        touches_damage_source = 1;

                    // Kill player
                    if (touches_damage_source)
                    {
                        p.alive = 0;
                    }
                }

                { // Mouse interaction
                    bool mouse_active = mouse.pos_delta().any() || mouse.left.down();
                    if (mouse_active)
                        w.cursor_idle_timer = 0;
                    else
                        w.cursor_idle_timer++;

                    w.cursor_pos = w.camera_pos.Value() + mouse.pos();

                    if (p.alive && mouse.left.pressed() && (abs(mouse.pos()) < screen_size/2).all())
                    {
                        ivec2 cursor_tile = Map::PixelToTilePos(w.cursor_pos);

                        bool at_button = 0;
                        map.ForEachTileLayer([&](Tiled::TileLayer &la)
                        {
                            if (Map::GetTile(cursor_tile, la) == Tile::button)
                                at_button = 1;
                        });

                        if (at_button)
                        {
                            ivec2 pixel_center = cursor_tile * tile_size + tile_size/2;

                            auto it = map.wire_elec_sources.find(cursor_tile);
                            if (it != map.wire_elec_sources.end())
                            {
                                map.wire_elec_sources.erase(it);
                                Sounds::power_on(pixel_center, 0.5);
                            }
                            else
                            {
                                map.wire_elec_sources.insert(cursor_tile);
                                Sounds::power_off(pixel_center, 0.5);
                            }

                            for (int i = 0; i < 14; i++)
                            {
                                constexpr fvec3 color_a = fvec3(0x58, 0x2e, 0x13) / 255, color_b = fvec3(0xfb, 0x96, 0x32) / 255;

                                par.AddFront(adjust(Particle{},
                                    pos = pixel_center + fvec2(-4 <= random.real() <= 4, -4 <= random.real() <= 4),
                                    vel = fvec2::dir(random.angle(), 0.05 <= random.real() <= 0.7),
                                    acc = fvec2(0,particle_gravity*2),
                                    current_time = 0,
                                    mid_time = 5,
                                    max_time = 20 <= random.integer() <= 40,
                                    size = 1 <= random.real() <= 3.5,
                                    color = random.boolean() ? color_b : color_a,
                                    beta = 0.75
                                ));
                            }

                            w.clicked_at_least_once = 1;
                            w.need_wire_update = 1;
                        }
                    }

                    if (w.clicked_at_least_once)
                        clamp_var_min(w.click_me_sign_alpha -= click_me_sign_alpha_step);
                }
            }

            { // Bullets
                auto it = w.bullets.begin();
                while (it != w.bullets.end())
                {
                    auto &bullet = *it;

                    bullet.pos.Offset(bullet.vel * subpixel_units);

                    if ((abs(bullet.pos.Value() - w.camera_pos.Value()) < screen_size/2 + 32).all())
                    {
                        for (int i = 0; i < 1; i++)
                        {
                            constexpr fvec3 color_a = fvec3(0xfb, 0x96, 0x32) / 255, color_b = fvec3(0xed, 0x07, 0x1e) / 255;

                            par.AddBack(adjust(Particle{},
                                pos = bullet.pos.Value() + fvec2(-1 <= random.real() <= 1, -1 <= random.real() <= 1),
                                vel = fvec2::dir(random.angle(), 0.05 <= random.real() <= 0.3) + bullet.vel * 0.8,
                                acc = fvec2(-sign(bullet.vel.x) * 0.1,0),
                                current_time = 0,
                                mid_time = 5,
                                max_time = 40 <= random.integer() <= 60,
                                size = 1 <= random.real() <= 6,
                                color = random.boolean() ? color_b : color_a,
                                beta = 0.75
                            ));
                        }
                    }

                    ivec2 point = bullet.pos.Value() with(x += sign(bullet.vel.x) * 5);
                    ivec2 point_tile = Map::PixelToTilePos(point);

                    if (map.HaveSolidTileAt(point_tile))
                    {
                        if (map.GetTileMid(point_tile) == Tile::target)
                        {
                            auto it = map.wire_elec_sources.find(point_tile);
                            if (it != map.wire_elec_sources.end())
                                map.wire_elec_sources.erase(it);
                            else
                                map.wire_elec_sources.insert(point_tile);
                            w.need_wire_update = 1;
                        }

                        if ((abs(bullet.pos.Value() - w.camera_pos.Value()) < screen_size/2 + 48).all())
                        {
                            for (int i = 0; i < 14; i++)
                            {
                                constexpr fvec3 color_a = fvec3(0xfb, 0x96, 0x32) / 255, color_b = fvec3(0xed, 0x07, 0x1e) / 255;

                                par.AddFront(adjust(Particle{},
                                    pos = point + fvec2(-1 <= random.real() <= 1, -1 <= random.real() <= 1),
                                    vel = fvec2::dir(random.angle(), 0.05 <= random.real() <= 0.7),
                                    acc = fvec2(0,particle_gravity),
                                    current_time = 0,
                                    mid_time = 5,
                                    max_time = 40 <= random.integer() <= 60,
                                    size = 1 <= random.real() <= 10,
                                    color = random.boolean() ? color_b : color_a,
                                    beta = 0.75
                                ));
                            }
                        }

                        Sounds::laser_hit(point);

                        it = w.bullets.erase(it);
                        continue;
                    }

                    if (w.final_box_hp > 0 && (abs(bullet.pos.Value() - w.final_box_pos) <= 13).all())
                    {
                        w.final_box_hp--;
                        if (w.final_box_hp == 0)
                        {
                            Program::Exit();
                        }

                        Sounds::box_hit(point);

                        it = w.bullets.erase(it);
                        continue;
                    }

                    bullet.age++;
                    if (bullet.age > 60 * 2)
                    {
                        it = w.bullets.erase(it);
                        continue;
                    }

                    it++;
                }
            }

            { // Powerups
                auto it = w.letter_powerups.begin();
                while (it != w.letter_powerups.end())
                {
                    auto &powerup = *it;

                    // Check for ground
                    bool old_ground = powerup.ground;
                    powerup.ground = powerup.hitbox.IsSolidAt(map, powerup.pos.Value() with(y += 1));

                    // Landing effects
                    if (!old_ground && powerup.ground)
                    {
                        Tile tile = map.GetTileMid(Map::PixelToTilePos(powerup.pos.Value() with(y += powerup.hitbox.half_extent.y + 2)));
                        if (tile == Tile::beam_h || tile == Tile::beam_v || tile == Tile::opt_block)
                            Sounds::land_metal();
                        else
                            Sounds::land();

                        for (int i = 0; i < 8; i++)
                        {
                            par.AddBack(adjust(Particle{},
                                pos = powerup.pos.Value() with(y += powerup.hitbox.half_extent.y),
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

                    // Apply gravity
                    powerup.vel_subpixel.y += gravity_subpixel;

                    // Clamp velocity
                    clamp_var(powerup.vel_subpixel, -ivec2(vel_cap_x, vel_cap_y), ivec2(vel_cap_x, vel_cap_y));

                    // Move
                    powerup.pos.Offset(powerup.vel_subpixel, [&](ivec2 new_pos)
                    {
                        return !powerup.hitbox.IsSolidAt(map, new_pos);
                    });

                    // Push out of walls
                    if (powerup.hitbox.IsSolidAt(map, powerup.pos.Value()))
                    {
                        // Try pushing player out of the wall first

                        bool in_wall = 1;
                        if (in_wall) // Vertical
                        {
                            for (int d = 1; d <= 8; d++)
                            {
                                for (int s = -1; s <= 1; s += 2)
                                {
                                    ivec2 new_pos = powerup.pos.Value() with(y += s * d);
                                    if (!powerup.hitbox.IsSolidAt(map, new_pos))
                                    {
                                        powerup.pos.Set(new_pos);
                                        in_wall = 0;
                                        break;
                                    }
                                }

                                if (!in_wall)
                                    break;
                            }
                        }
                        if (in_wall) // Horizontal
                        {
                            for (int d = 1; d <= 6; d++)
                            {
                                for (int s = -1; s <= 1; s += 2)
                                {
                                    ivec2 new_pos = powerup.pos.Value() with(x += s * d);
                                    if (!powerup.hitbox.IsSolidAt(map, new_pos))
                                    {
                                        powerup.pos.Set(new_pos);
                                        in_wall = 0;
                                        break;
                                    }
                                }

                                if (!in_wall)
                                    break;
                            }
                        }
                    }

                    // Set velocity to null on impact
                    for (int i = 0; i < 2; i++)
                    {
                        auto &vel_i = powerup.vel_subpixel[i];
                        ivec2 dir{};
                        dir[i] = sign(vel_i);

                        if (powerup.hitbox.IsSolidAt(map, powerup.pos.Value() + dir))
                            vel_i = 0;
                    }

                    // Player interaction
                    {
                        bool player_has_this_letter = p.enabled_controls.count(powerup.ch);

                        if (!player_has_this_letter)
                        {
                            if ((abs(powerup.pos.Value() - p.pos.Value()) <= 8).all())
                            {
                                for (int i = 0; i < 20; i++)
                                {
                                    par.AddBack(adjust(Particle{},
                                        pos = powerup.pos.Value() + ivec2(-7 <= random.real() <= 7, -7 <= random.real() <= 7),
                                        vel = fvec2::dir(random.angle(), 0 <= random.real() <= 0.3),
                                        acc = fvec2(0,particle_gravity),
                                        current_time = 0,
                                        mid_time = 5,
                                        max_time = 40 <= random.integer() <= 60,
                                        size = 1 <= random.real() <= 5,
                                        color = fvec3(0.7 <= random.real() <= 0.9)
                                    ));
                                }

                                Sounds::powerup(powerup.pos.Value(), 0.5);

                                p.enabled_controls.insert(powerup.ch);

                                it = w.letter_powerups.erase(it);
                                continue;
                            }
                        }
                    }

                    it++;
                }
            }

            { // Checkpoints
                auto it = w.checkpoints.begin();
                while (it != w.checkpoints.end())
                {
                    ivec2 checkpoint = *it;

                    if ((abs(checkpoint - w.camera_pos.Value()) < screen_size/2 + 32).all() && (0 <= random.integer() <= 4) == 0)
                    {
                        par.AddFront(adjust(Particle{},
                            pos = checkpoint + ivec2(-3 <= random.real() <= 3, -3 <= random.real() <= 3),
                            vel = fvec2::dir(random.angle(), 0 <= random.real() <= 0.3),
                            current_time = 0,
                            mid_time = 5,
                            max_time = 40 <= random.integer() <= 60,
                            size = 2 <= random.real() <= 6,
                            color = fvec3(0.5 <= random.real() <= 0.75, 1, 0)
                        ));
                    }

                    // Collect
                    if (((p.pos.Value() - checkpoint).abs() < p.hitbox.half_extent + 3).all())
                    {
                        for (int i = 0; i < 20; i++)
                        {
                            par.AddFront(adjust(Particle{},
                                pos = checkpoint + ivec2(-3 <= random.real() <= 3, -3 <= random.real() <= 3),
                                vel = fvec2::dir(random.angle(), 0 <= random.real() <= 0.5),
                                acc = fvec2(0,particle_gravity),
                                current_time = 0,
                                mid_time = 5,
                                max_time = 40 <= random.integer() <= 60,
                                size = 1 <= random.real() <= 4,
                                color = fvec3(0.5 <= random.real() <= 0.75, 1, 0)
                            ));
                        }

                        Sounds::checkpoint();

                        it = w.checkpoints.erase(it);

                        saved_game = *this;

                        continue;
                    }

                    it++;
                }
            }

            { // Camera
                if (p.alive)
                {
                    fvec2 delta = (w.camera_pos.SubpixelValue() - p.pos.SubpixelValue()) / float(subpixel_units) + ivec2(0, camera_y_offset);
                    float dist = delta.len_sqr();
                    if (dist > 1)
                    {
                        dist = sqrt(dist);

                        fvec2 acc = -delta / dist * std::pow(dist / camera_ref_dist, camera_acc_pow) / camera_mass;
                        w.camera_vel += acc;

                        w.camera_pos.Offset(w.camera_vel * subpixel_units);
                    }

                    w.camera_vel *= camera_drag_fac;
                }
                else
                {
                    w.camera_pos.Offset(w.camera_vel * subpixel_units);
                }

                Audio::Listener::Position(w.camera_pos.Value().to_vec3(-3 * screen_size.x/2));
                Audio::Listener::Orientation(fvec3(0,0,1), fvec3(0,-1,0));
            }

            { // Particles
                par.Tick();
            }

            { // Player death (keep near the end of tick)
                if (!p.alive)
                {
                    // Death effects
                    if (p.death_timer == 0)
                    {
                        Sounds::death(2);

                        for (int i = 0; i < 30; i++)
                        {
                            float t = 0 <= random.real() <= 1;

                            par.AddFront(adjust(Particle{},
                                pos = p.pos.Value() + fvec2(-4 <= random.real() <= 4, -8 <= random.real() <= 8),
                                vel = fvec2::dir(random.angle(), 0.05 <= random.real() <= 0.8),
                                acc = fvec2(0,-particle_gravity),
                                current_time = 0,
                                mid_time = 5,
                                max_time = 60 <= random.integer() <= 90,
                                size = 1 <= random.real() <= 8,
                                color = (fmat3::rotate(fvec3(1,1,1), (t*2-1) * f_pi / 6) * fvec3(1,0,0)) * (0.5 + t * 0.5),
                                beta = 0.75
                            ));
                        }
                    }

                    // Death screen fade
                    if (p.death_timer > 20)
                    {
                        w.death_fade_out += death_fade_step;
                        if (w.death_fade_out > 1.2)
                        {
                            ivec2 old_cam_pos = w.camera_pos.Value();
                            *this = saved_game;
                            par.Reset();
                            w.camera_pos.Set(old_cam_pos);
                            w.camera_vel = fvec2(0);
                            w.fade_in = 1;
                            return;
                        }
                    }

                    p.death_timer++;
                }
            }

            { // Misc timers
                clamp_var_min(w.fade_in -= fade_step);

                if (w.pressed_at_least_once)
                    clamp_var_min(w.key_enabled_sign_alpha -= click_me_sign_alpha_step);

                Sounds::theme_src.volume((w.final_box_hp / float(w.final_box_max_hp)) * 0.3);
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

            // Map
            map.RenderLayerBack(w.camera_pos.Value(), false);
            map.RenderLayerBack(w.camera_pos.Value(), true);
            map.RenderLayerMid(w.camera_pos.Value(), false);

            { // Powerups
                for (const LetterPowerup &powerup : w.letter_powerups)
                {
                    r << r.translate(powerup.pos.Value() - w.camera_pos.Value())
                       * r.TexturedQuad(sprites.letters.region(ivec2(16 * powerup.TexIndex(), 0),ivec2(16))).Centered();
                }
            }

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

                float alpha = clamp_min(1 - p.death_timer / 15.);

                constexpr ivec2 player_sprite_size(24);
                r << r.translate(p.pos.Value() - w.camera_pos.Value() + ivec2(0,-1)) * r.flip_x(p.facing_left)
                   * r.TexturedQuad(sprites.player.region(player_sprite_size * ivec2(anim_frame, anim_state), player_sprite_size)).Centered().Opacity(alpha);
            }

            // Final box
            r << r.translate(w.final_box_pos - w.camera_pos.Value())
               * r.TexturedQuad(sprites.final_box).Centered();

            // Particles
            par.Render(w.camera_pos.Value(), false);

            { // Bullets
                for (const Bullet &bullet : w.bullets)
                {
                    r << r.translate(bullet.pos.Value() - w.camera_pos.Value()) * r.flip_x(bullet.vel.x < 0)
                       * r.TexturedQuad(sprites.bullet).Centered();
                }
            }

            { // Checkpoints
                for (ivec2 pos : w.checkpoints)
                {
                    fvec3 color = fvec3(0.5 <= random.real() <= 0.75, 1, 0);

                    r << r.translate(pos - w.camera_pos.Value()) * r.rotate(f_pi / 4)
                       * r.UntexturedQuad(ivec2(9)).Centered().Color(color / 3);

                    int size = 7 + iround(sin(window.Ticks() % 20 / 20.f * 2 * f_pi));
                    r << r.translate(pos - w.camera_pos.Value()) * r.rotate(f_pi / 4)
                       * r.UntexturedQuad(ivec2(size)).Centered().Color(color);
                }
            }

            // Map (front)
            map.RenderLayerMid(w.camera_pos.Value(), true);
            map.RenderLayerWire(w.camera_pos.Value(), false);
            map.RenderLayerWire(w.camera_pos.Value(), true);

            // Particles (front)
            par.Render(w.camera_pos.Value(), true);

            // Controls
            {
                static constexpr ivec2 base_pos = screen_size / ivec2(-2,2) + ivec2(10,-10);

                for (int i = 0; i < 4; i++)
                {
                    bool key_down = (i == 0 ? controls.left :
                                     i == 1 ? controls.jump :
                                     i == 2 ? controls.right :
                                              controls.shoot).down();
                    bool key_enabled = p.enabled_controls.count("awdj"[i]);

                    int state = 0;

                    if (key_enabled)
                    {
                        if (key_down)
                            state = 2;
                        else
                            state = 1;
                    }

                    r << r.translate(base_pos + ivec2(i*10 + (i >= 3) * 5, i == 1 ? -5 : 0))
                       * r.TexturedQuad(sprites.small_letters.region(ivec2(9 * i + 36 * state, 0), ivec2(9))).CenterRel(fvec2(0,1)).Color(fvec3(0), !key_enabled && key_down);
                }

                // "Esc" button
                r << r.translate(base_pos + ivec2(50, 0))
                   * r.TexturedQuad(sprites.small_letters.region(ivec2(108,0), ivec2(21,9))).CenterRel(fvec2(0,1));

                // "Key enabled" sign
                if (w.key_enabled_sign_alpha > 0.001 && p.enabled_controls.size() > 0)
                {
                    r << r.translate(base_pos + ivec2(30, -10))
                       * r.TexturedQuad(sprites.key_enabled).CenterRel(fvec2(0,1)).Opacity(w.key_enabled_sign_alpha - 0.2 * (sin(window.Ticks() % 30 / 30.f * 2 * f_pi) * 0.5 + 0.5));
                }
            }

            { // Cursor
                // "Click me" sign and author info
                if (w.click_me_sign_alpha > 0.001)
                {
                    r << r.translate(map.click_me_sign - w.camera_pos.Value())
                       * r.TexturedQuad(sprites.click_me).CenterTex(ivec2(15,2)).Opacity(w.click_me_sign_alpha - 0.2 * (sin(window.Ticks() % 30 / 30.f * 2 * f_pi) * 0.5 + 0.5));

                    Draw::Text(adjust(Draw::TextData{}, pos = ivec2(0,-screen_size.y/2 - 3), align = ivec2(0,-1), color = fvec3(0), alpha = w.click_me_sign_alpha * 0.75,
                        text = Graphics::Text(font_main, "WADJ - a game by HolyBlackCat, made for LD45 in Oct 2019")));
                }

                if (w.cursor_idle_timer < 90)
                {
                    // Tile frame
                    ivec2 mouse_tile = Map::PixelToTilePos(w.cursor_pos);
                    if (map.GetTileWire(mouse_tile) == Tile::button)
                        r << r.translate(mouse_tile * tile_size + tile_size/2 - w.camera_pos.Value()) * r.TexturedQuad(sprites.frame).Centered();

                    // Actual cursor
                    r << r.translate(mouse.pos()) * r.TexturedQuad(sprites.cursor).Centered();
                }
            }

            // Vignette
            r << r.TexturedQuad(sprites.vignette).Centered();

            { // Screen fade
                { // Finale
                    if (w.final_box_hp < w.final_box_max_hp)
                    {
                        float t = 1 - (w.final_box_hp - 1) / float(w.final_box_max_hp);
                        // t = std::pow(t, 0.85);

                        fvec3 color(1,0.9,0.8);

                        r << r.UntexturedQuad(screen_size).Color(color * (1-t) + fvec3(1) * t).Centered().Opacity(t, t);

                        static constexpr float text_th = 0.75;

                        if (t > text_th)
                        {
                            Draw::Text(adjust(Draw::TextData{}, color = fvec3(0), alpha = (t - text_th) / (1 - text_th), pos = fvec2(0), text = Graphics::Text(font_main, "Thanks for playing! <3")));
                        }
                    }
                }

                { // Death fade
                    float t = clamp(w.death_fade_out);

                    if (t > 0.001)
                    {
                        float t_acc = std::pow(t, 2);

                        // Light
                        r << r.UntexturedQuad(screen_size).Centered().Color(fvec3(1,0.9,0.8)).Opacity(t_acc * 1, 1);

                        // Black strips
                        r << r.translate(ivec2(0, -(1 - t_acc) * screen_size.y / 2))
                           * r.UntexturedQuad(screen_size with(y /= 2)).CenterRel(fvec2(0.5,1)).Color(fvec3(0));
                        r << r.translate(ivec2(0, (1 - t_acc) * screen_size.y / 2))
                           * r.UntexturedQuad(screen_size with(y /= 2)).CenterRel(fvec2(0.5,0)).Color(fvec3(0));
                    }
                }

                { // Respawn fade
                    if (w.fade_in > 0.001)
                        r << r.UntexturedQuad(screen_size).Centered().Color(fvec3(0)).Opacity(w.fade_in, 1);
                }
            }


            r.Flush();
        }
    };

    Game Game::saved_game;
}

int ENTRY_POINT(int, char **)
{
    window.SetMode(Interface::borderless_fullscreen);

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

            gui_controller.LoadFont("assets/CatIV15.ttf", 15.0f, adjust(ImFontConfig{}, RasterizerFlags = monochrome_font_flags));
            gui_controller.LoadDefaultFont();
            gui_controller.RenderFontsWithFreetype();

            ImGui::GetIO().MouseDrawCursor = 1;
        }

        { // Audio
            Audio::Source::DefaultRefDistance(4 * screen_size.x / 2);
            Audio::Source::DefaultMaxDistance(4 * screen_size.x / 2);
            Audio::Source::DefaultRolloffFactor(1);

            Sounds::theme_src = Audio::Source(Sounds::theme_buffer).loop(1).play();
        }

        mouse.HideCursor();
    }

    auto Resize = [&]
    {
        adaptive_viewport.Update();
        r.SetMatrix(adaptive_viewport.GetDetails().MatrixCentered());
        mouse.SetMatrix(adaptive_viewport.GetDetails().MouseMatrixCentered());
    };
    Resize();

    States::current_state.assign<States::Game>().Init();

    Metronome metronome(60);
    Clock::DeltaTimer delta_timer;

    while (1)
    {
        uint64_t delta = delta_timer();
        while (metronome.Tick(delta))
        {
            // window.ProcessEvents();
            window.ProcessEvents({gui_controller.EventHook()});

            if ((Input::Button(Input::l_alt).down() || Input::Button(Input::r_alt).down()) && Input::Button(Input::enter).pressed())
            {
                window.SetMode(window.Mode() == Interface::windowed ? Interface::borderless_fullscreen : Interface::windowed);
            }

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
        if (ImGui::IsAnyWindowHovered())
            gui_controller.PostRender();

        window.SwapBuffers();
    }
}
