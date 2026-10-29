import napari
from magicgui.widgets import ComboBox
from ferroscope.colors import COLOR_KEYS, colormap_list, default_color, parse_color
from ferroscope.io import load_scene
from ferroscope.viewer import SceneBrowser


def ask_colors(channel_names: list[str]) -> dict[str, str]:
    hint = "/".join(COLOR_KEYS)
    colors = {}
    for i, name in enumerate(channel_names):
        default = default_color(i)
        while True:
            answer = input(f"{name} color [{hint}] (Enter = {default}): ")
            try:
                colors[name] = parse_color(answer, default)
                break
            except ValueError as e:
                print(e)
    return colors


path = "~/data/OGD_but/PL1_live.czi"

first = load_scene(path, 0)                  # only to learn the channel names
colors = ask_colors(first.channel_names)

viewer = napari.Viewer()
browser = SceneBrowser(viewer, path, colors)
browser.go(0)

# Step 2: dropdown
combo = ComboBox(choices=browser.scenes, label="Scene")
combo.changed.connect(browser.go_to)
viewer.window.add_dock_widget(combo, area="right", name="Scenes")

# Step 3: PageDown / PageUp move the dropdown, so both stay in sync
def step(delta: int) -> None:
    i = (browser.index + delta) % len(browser.scenes)
    combo.value = browser.scenes[i]

viewer.bind_key("PageDown", lambda v: step(+1), overwrite=True)
viewer.bind_key("PageUp", lambda v: step(-1), overwrite=True)

napari.run()
