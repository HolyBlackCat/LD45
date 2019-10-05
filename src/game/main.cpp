constexpr ivec2 screen_size(480, 270);
Interface::Window window("LD45", screen_size * 2, Interface::windowed, adjust_(Interface::WindowSettings{}, min_size = screen_size));
Graphics::DummyVertexArray dummy_vao = nullptr;

Input::Mouse mouse;

const Graphics::ShaderConfig shader_config = Graphics::ShaderConfig::Core();
Interface::ImGuiController gui_controller(Poly::derived<Interface::ImGuiController::GraphicsBackend_Modern>, adjust_(Interface::ImGuiController::Config{}, shader_header = shader_config.common_header));

Graphics::TextureAtlas texture_atlas(ivec2(2048), "assets/_images", "assets/atlas.png", "assets/atlas.refl");
Graphics::Texture texture_main = Graphics::Texture(nullptr).Wrap(Graphics::clamp).Interpolation(Graphics::nearest);

AdaptiveViewport adaptive_viewport(shader_config, screen_size);

using Renderer = Graphics::Renderers::Flat;
Renderer r(shader_config, 1000);

ReflectStruct(Sprites, (
    (Graphics::TextureAtlas::Region)(font_storage),
))
Sprites sprites;

Graphics::Font font_main;
Graphics::FontFile font_file_main("assets/CatIV15.ttf", 15);

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
        Game()
        {

        }

        void Tick() override
        {
            // ImGui::ShowDemoWindow();
        }

        void Render() const override
        {
            Graphics::Clear();

            r << r.UntexturedQuad(screen_size).Color(fvec3(0,0,0.1) / 4).Centered();
            Draw::Text(adjust(Draw::TextData{}, text = Graphics::Text(font_main, "Hello world!\nHello!"), pos = mouse.pos(), color = fvec3(1,0.5,0)));

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
