"""Full 56-asset manifest and prompt specifications for Van Kiem Quy Tong."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CHECKPOINT_NAME = "animagine-xl-3.1.safetensors"
LORA_NAME = "[XL]Sumi-e.safetensors"
LORA_WEIGHT = 0.75

NEGATIVE_PROMPT = (
    "3d render, realistic, photorealistic, blurry, low quality, watermark, "
    "signature, text, english letters, chinese characters, multiple views, "
    "character sheet, grid, split screen, cropped, out of frame, "
    "messy background, grey background, gradient background, drop shadow"
)

AssetMode = Literal[
    "sprite_bottom",
    "icon_center",
    "ui_panel",
    "ui_border",
    "ui_bar_frame",
    "ui_bar_fill",
    "opaque_center",
    "opaque_scene",
]


@dataclass(frozen=True)
class AssetSpec:
    """Specification for a single generated game asset."""

    id: str
    category: str
    subfolder: str
    prompt: str
    target_size: tuple[int, int]
    remove_bg: bool = True
    gen_size: tuple[int, int] = (1024, 1024)
    mode: AssetMode = "icon_center"
    min_area_ratio: float = 0.10
    max_area_ratio: float = 0.88
    max_fragments: int = 10
    max_debris_ratio: float = 0.18
    primary_parts: int = 1
    padding_ratio: float = 0.04


def _char(asset_id: str, prompt: str) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category="Characters",
        subfolder="Characters",
        prompt=prompt,
        target_size=(512, 512),
        remove_bg=True,
        mode="sprite_bottom",
        min_area_ratio=0.10,
        max_fragments=8,
        max_debris_ratio=0.15,
        padding_ratio=0.03,
    )


def _enemy(asset_id: str, prompt: str, max_fragments: int = 10) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category="Enemies",
        subfolder="Enemies",
        prompt=prompt,
        target_size=(512, 512),
        remove_bg=True,
        mode="sprite_bottom",
        min_area_ratio=0.10,
        max_fragments=max_fragments,
        max_debris_ratio=0.18,
        padding_ratio=0.03,
    )


def _boss(asset_id: str, prompt: str, max_fragments: int = 16) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category="Boss",
        subfolder="Enemies",
        prompt=prompt,
        target_size=(1024, 1024),
        remove_bg=True,
        mode="sprite_bottom",
        min_area_ratio=0.10,
        max_fragments=max_fragments,
        max_debris_ratio=0.20,
        padding_ratio=0.03,
    )


def _weapon(asset_id: str, prompt: str) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category="Weapons",
        subfolder="Icons",
        prompt=prompt,
        target_size=(256, 256),
        remove_bg=True,
        mode="icon_center",
        min_area_ratio=0.10,
        max_fragments=10,
        max_debris_ratio=0.18,
        padding_ratio=0.06,
    )


def _elixir(asset_id: str, color_desc: str) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category="DanDuoc",
        subfolder="Icons",
        prompt=(
            f"ancient chinese porcelain medicine gourd bottle next to a floating glowing "
            f"{color_desc} elixir pill with swirling spiritual mist, traditional sumi-e "
            f"ink wash painting style, high contrast, isolated on pure white background, "
            f"2d game item icon, centered"
        ),
        target_size=(256, 256),
        remove_bg=True,
        mode="icon_center",
        min_area_ratio=0.10,
        max_fragments=12,
        max_debris_ratio=0.22,
        primary_parts=2,
        padding_ratio=0.06,
    )


def _fruit(asset_id: str, fruit_desc: str) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category="DaoQua",
        subfolder="Icons",
        prompt=(
            f"mythical immortal heavenly fruit {fruit_desc} radiating mystical aura and "
            f"ink smoke, traditional chinese sumi-e ink painting style, high contrast, "
            f"isolated on pure white background, 2d game icon, centered"
        ),
        target_size=(256, 256),
        remove_bg=True,
        mode="icon_center",
        min_area_ratio=0.10,
        max_fragments=12,
        max_debris_ratio=0.20,
        padding_ratio=0.06,
    )


def _tag(asset_id: str, emblem_phrase: str) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category="Tags",
        subfolder="Icons",
        prompt=(
            f"minimalist circular chinese calligraphy seal icon featuring {emblem_phrase}, "
            f"black ink brush stroke on pure white background, high contrast sumi-e style, "
            f"2d game UI badge"
        ),
        target_size=(256, 256),
        remove_bg=True,
        mode="icon_center",
        min_area_ratio=0.10,
        max_fragments=14,
        max_debris_ratio=0.25,
        padding_ratio=0.06,
    )


def _pickup(
    asset_id: str,
    prompt: str,
    subfolder: str = "Icons",
    category: str = "Pickups",
    target_size: tuple[int, int] = (128, 128),
) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category=category,
        subfolder=subfolder,
        prompt=prompt,
        target_size=target_size,
        remove_bg=True,
        mode="icon_center",
        min_area_ratio=0.10,
        max_fragments=12,
        max_debris_ratio=0.22,
        padding_ratio=0.06,
    )


def _border(asset_id: str, glow_color: str) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            f"square 2d game item slot border frame with glowing {glow_color} ink brush "
            f"energy aura, completely empty transparent-ready white center, ornate oriental "
            f"corner flourishes, flat 2d game UI border, isolated on pure white background"
        ),
        target_size=(256, 256),
        remove_bg=True,
        mode="ui_border",
        min_area_ratio=0.10,
        max_area_ratio=0.85,
        max_fragments=14,
        max_debris_ratio=0.25,
        padding_ratio=0.02,
    )


ASSETS: tuple[AssetSpec, ...] = (
    # --- 4 Characters (512x512 -> Characters/) ---
    _char(
        "Char_KiemTu",
        "solo, full body, young taoist swordsman cultivator in flowing white and black "
        "ink wash robes with glowing cyan trim, glowing cyan eyes, empty hands in "
        "dynamic martial arts idle stance, no weapon in hand, traditional chinese "
        "sumi-e ink painting style, high contrast, isolated on pure white background, "
        "clean edges, 2d game character sprite",
    ),
    _char(
        "Char_TheTu",
        "solo, full body, muscular martial monk body cultivator with wrapped cloth "
        "bandages on fists and torso, dark ink wash trousers, glowing earthy gold aura "
        "and eyes, powerful grounded stance, traditional chinese sumi-e ink painting "
        "style, high contrast, isolated on pure white background, clean edges, "
        "2d game character sprite",
    ),
    _char(
        "Char_PhuTu",
        "solo, full body, taoist master talisman cultivator wearing traditional scholar "
        "crown hat and flowing ink wash robes, holding a horsetail whisk, glowing "
        "azure blue aura and eyes, mystic pose, traditional chinese sumi-e ink "
        "painting style, high contrast, isolated on pure white background, clean "
        "edges, 2d game character sprite",
    ),
    _char(
        "Char_DuocTu",
        "solo, full body, ancient eastern alchemist healer cultivator wearing scholar "
        "robes with a large medicine gourd tied on back, glowing emerald green aura "
        "and eyes, calm agile pose, traditional chinese sumi-e ink painting style, "
        "high contrast, isolated on pure white background, clean edges, "
        "2d game character sprite",
    ),
    # --- 5 Enemies & 2 Boss (Enemies/) ---
    _enemy(
        "Enemy_HacLang",
        "solo, demonic black shadow wolf monster made of swirling dark ink smoke, "
        "fiercely glowing emerald green eyes and forest spirit markings, aggressive "
        "pouncing pose, traditional chinese sumi-e ink painting style, high contrast, "
        "isolated on pure white background, clean edges, 2d game enemy sprite",
    ),
    _enemy(
        "Enemy_HuyetDoi",
        "solo, monstrous demonic bat with tattered black ink wings, bloated belly "
        "glowing bright crimson red like molten fire about to explode, glowing red "
        "eyes, flying pose, traditional chinese sumi-e ink painting style, high "
        "contrast, isolated on pure white background, clean edges, 2d game enemy sprite",
    ),
    _enemy(
        "Enemy_DocGiacXa",
        "solo, giant venomous single-horned serpent monster coiled and ready to strike, "
        "dark ink black scales with glowing sapphire blue bioluminescent patterns and "
        "eyes, traditional chinese sumi-e ink painting style, high contrast, isolated "
        "on pure white background, clean edges, 2d game enemy sprite",
    ),
    _enemy(
        "Enemy_KhoCot",
        "solo, undead skeleton shaman sorcerer wearing tattered black ink robes, "
        "holding a gnarled bone staff, glowing amber brown earth energy and eyes, "
        "traditional chinese sumi-e ink painting style, high contrast, isolated on "
        "pure white background, clean edges, 2d game enemy sprite",
    ),
    _enemy(
        "Enemy_ThietGiapHung",
        "solo, colossal armored bear monster wearing heavy ancient dark iron plates, "
        "glowing violet purple metallic aura and eyes, intimidating towering stance, "
        "traditional chinese sumi-e ink painting style, high contrast, isolated on "
        "pure white background, clean edges, 2d game elite enemy sprite",
    ),
    _boss(
        "Boss_HuyetNha",
        "solo, colossal demonic wild boar demon king, massive black ink fur body "
        "covered in battle scars, huge razor-sharp tusks glowing bright crimson red, "
        "ferocious roaring pose, billowing dark ink smoke, traditional chinese sumi-e "
        "ink painting style, high contrast, isolated on pure white background, clean "
        "edges, 2d game boss sprite",
    ),
    _boss(
        "Boss_CuuULoiGiao",
        "solo, colossal abyssal black ink flood dragon coiling in the air, surrounded "
        "by crackling glowing purple lightning bolts, fierce horns and glowing violet "
        "eyes, majestic terrifying boss pose, traditional chinese sumi-e ink painting "
        "style, high contrast, isolated on pure white background, clean edges, "
        "2d game boss sprite",
        max_fragments=18,
    ),
    # --- 6 Weapons (256x256 -> Icons/) ---
    _weapon(
        "Weapon_CuKiem",
        "vertical heavy black iron broadsword carved with glowing golden ancient "
        "runes, massive blade, traditional chinese sumi-e ink painting style, high "
        "contrast, isolated on pure white background, 2d game weapon icon, centered, "
        "clean edges",
    ),
    _weapon(
        "Weapon_KimChung",
        "ancient bronze Taoist temple bell engraved with glowing golden Bagua trigrams, "
        "mystical protective aura, traditional chinese sumi-e ink painting style, high "
        "contrast, isolated on pure white background, 2d game weapon icon, centered, "
        "clean edges",
    ),
    _weapon(
        "Weapon_HuyetDao",
        "vertical curved demonic saber blade dripping with glowing crimson blood mist, "
        "dark ink steel, traditional chinese sumi-e ink painting style, high contrast, "
        "isolated on pure white background, 2d game weapon icon, centered, clean edges",
    ),
    _weapon(
        "Weapon_LienHoaDang",
        "ornate ancient lotus-shaped palace lantern burning with a glowing cyan-white "
        "spiritual soul flame, traditional chinese sumi-e ink painting style, high "
        "contrast, isolated on pure white background, 2d game weapon icon, centered, "
        "clean edges",
    ),
    _weapon(
        "Weapon_ThanhTrucKiem",
        "vertical slender agile flying sword crafted from glowing emerald green jade "
        "bamboo, sharp ethereal edge, traditional chinese sumi-e ink painting style, "
        "high contrast, isolated on pure white background, 2d game weapon icon, "
        "centered, clean edges",
    ),
    _weapon(
        "Weapon_HuyenBangPhu",
        "vertical ancient Taoist paper talisman infused with glowing icy sapphire blue "
        "frost runes and ice crystals, traditional chinese sumi-e ink painting style, "
        "high contrast, isolated on pure white background, 2d game weapon icon, "
        "centered, clean edges",
    ),
    # --- 6 Dan Duoc (256x256 -> Icons/) ---
    _elixir("Item_Dan_HoiHuyet", "crimson red"),
    _elixir("Item_Dan_TuKhi", "emerald green"),
    _elixir("Item_Dan_BaoNguyen", "golden"),
    _elixir("Item_Dan_TayTuy", "azure blue"),
    _elixir("Item_Dan_HoTam", "violet"),
    _elixir("Item_Dan_PhiThang", "pure white"),
    # --- 6 Dao Qua (256x256 -> Icons/) ---
    _fruit("Fruit_KiemTam", "glowing golden peach"),
    _fruit("Fruit_BatDiet", "fiery red dragon fruit"),
    _fruit("Fruit_LoiDinh", "icy blue spirit ginseng fruit"),
    _fruit("Fruit_HonDon", "purple lightning berry"),
    _fruit("Fruit_TruongSinh", "emerald jade pear"),
    _fruit("Fruit_ThonPhe", "yin-yang black and white fruit"),
    # --- 5 Tags (256x256 -> Icons/) ---
    _tag("Tag_KiemDao", "crossed flying swords"),
    _tag("Tag_TheDao", "a glowing fist"),
    _tag("Tag_PhuDao", "a mystic talisman scroll"),
    _tag("Tag_DanDao", "an alchemy gourd"),
    _tag("Tag_MaDao", "a demonic skull flame"),
    # --- 3 Pickups (128x128 -> Icons/) & 3 Environment (Environment/) ---
    _pickup(
        "Pickup_LinhKhi",
        "small glowing cyan-blue spiritual energy orb crystal radiating soft light, "
        "minimalist black ink outline, sumi-e style, isolated on pure white "
        "background, 2d game pickup icon, centered",
        subfolder="Icons",
        category="Pickups",
        target_size=(128, 128),
    ),
    _pickup(
        "Pickup_LinhThach",
        "faceted glowing golden spirit stone gem crystal with sharp edges, minimalist "
        "black ink outline, sumi-e style, isolated on pure white background, 2d game "
        "currency icon, centered",
        subfolder="Icons",
        category="Pickups",
        target_size=(128, 128),
    ),
    _pickup(
        "Pickup_RuongBau",
        "ancient dark wooden Chinese treasure chest reinforced with glowing golden trim "
        "and lock, black ink wash texture, sumi-e style, isolated on pure white "
        "background, 2d game icon, centered",
        subfolder="Icons",
        category="Pickups",
        target_size=(128, 128),
    ),
    _pickup(
        "VFX_CanhBao_X",
        "bold aggressive hand-painted calligraphy brush stroke X mark in glowing "
        "crimson red ink, splatter edges, sumi-e style, isolated on pure white "
        "background, 2d game warning marker, centered",
        subfolder="Environment",
        category="Environment",
        target_size=(128, 128),
    ),
    AssetSpec(
        id="Env_ArenaFloor",
        category="Environment",
        subfolder="Environment",
        prompt=(
            "top-down flat 2d game arena floor texture, dark charcoal-grey ancient "
            "xuan paper with subtle faded black ink wash clouds and a faint mystical "
            "Bagua formation circle in the center, seamless dark background"
        ),
        target_size=(1024, 1024),
        remove_bg=False,
        gen_size=(1024, 1024),
        mode="opaque_center",
        padding_ratio=0.0,
    ),
    _pickup(
        "Env_StonePillar",
        "ancient broken chinese stone pillar wrapped in dark ink mist and faint "
        "talisman carvings, traditional sumi-e ink painting style, high contrast, "
        "isolated on pure white background, 2d game environment prop",
        subfolder="Environment",
        category="Environment",
        target_size=(256, 256),
    ),
    # --- 16 UI_Kit (UI_Kit/) ---
    AssetSpec(
        id="UI_Panel_Scroll",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "rectangular ancient dark charcoal-grey parchment scroll UI window frame "
            "with subtle glowing muted gold ink brush borders and corner cloud motifs, "
            "empty dark center, top-down flat 2d game UI panel, isolated on pure white "
            "background, clean straight edges"
        ),
        target_size=(512, 512),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="ui_panel",
        min_area_ratio=0.15,
        max_area_ratio=0.92,
        max_fragments=10,
        max_debris_ratio=0.18,
        padding_ratio=0.02,
    ),
    AssetSpec(
        id="UI_Card_Frame",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "vertical rectangular card frame for 2d game UI, dark sumi-e ink wash "
            "background texture, ornate silver-white calligraphy brush border, empty "
            "center space for icon and text, flat 2d game UI asset, isolated on pure "
            "white background"
        ),
        target_size=(256, 512),
        remove_bg=True,
        gen_size=(704, 1408),
        mode="ui_panel",
        min_area_ratio=0.15,
        max_area_ratio=0.92,
        max_fragments=10,
        max_debris_ratio=0.18,
        padding_ratio=0.02,
    ),
    _border("Border_1_Pham", "pure white"),
    _border("Border_2_Linh", "azure blue"),
    _border("Border_3_Dia", "violet purple"),
    _border("Border_4_Thien", "golden orange"),
    _border("Border_5_Tien", "blazing crimson red"),
    AssetSpec(
        id="UI_Bar_Frame",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "horizontal elongated rectangular 2d game health bar frame, dark ink iron "
            "border with sharp calligraphy brush ends, empty white interior, flat 2d "
            "game UI asset, isolated on pure white background"
        ),
        target_size=(512, 64),
        remove_bg=True,
        gen_size=(1536, 640),
        mode="ui_bar_frame",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=10,
        max_debris_ratio=0.22,
        padding_ratio=0.02,
    ),
    AssetSpec(
        id="UI_Bar_Fill_White",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "horizontal solid pure white bar fill with subtle light grey ink wash "
            "texture, flat 2d game UI asset, isolated on black background"
        ),
        target_size=(512, 64),
        remove_bg=False,
        gen_size=(1536, 640),
        mode="ui_bar_fill",
        padding_ratio=0.0,
    ),
    AssetSpec(
        id="UI_Icon_DonThuat",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "mystical glowing cyan-white crane feather infused with wind ink trails, "
            "sumi-e brush style, isolated on pure white background, 2d game skill icon, "
            "centered"
        ),
        target_size=(128, 128),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_fragments=12,
        max_debris_ratio=0.20,
        padding_ratio=0.06,
    ),
    AssetSpec(
        id="UI_Button_Frame",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "horizontal rectangular 2d game UI button frame, dark ink iron texture "
            "with subtle glowing gold calligraphy brush border, empty dark center, "
            "flat 2d game UI asset, isolated on pure white background"
        ),
        target_size=(512, 128),
        remove_bg=True,
        gen_size=(1536, 640),
        mode="ui_bar_frame",
        min_area_ratio=0.12,
        max_area_ratio=0.90,
        max_fragments=10,
        max_debris_ratio=0.20,
        padding_ratio=0.02,
    ),
    AssetSpec(
        id="UI_MainMenu_BG",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "breathtaking dark chinese sumi-e ink wash landscape painting of misty "
            "floating mountain peaks and ancient taoist temples at night, glowing cyan "
            "moonlight, atmospheric 2d game main menu background, no text"
        ),
        target_size=(1920, 1080),
        remove_bg=False,
        gen_size=(1344, 768),
        mode="opaque_scene",
        padding_ratio=0.0,
    ),
    AssetSpec(
        id="UI_Game_Logo",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "ornate ancient chinese calligraphy ink brush crest with crossed glowing "
            "cyan flying swords and mystical cloud seal emblem, center space for "
            "title, sumi-e style, isolated on pure white background"
        ),
        target_size=(1024, 512),
        remove_bg=True,
        gen_size=(1408, 704),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=14,
        max_debris_ratio=0.22,
        padding_ratio=0.04,
    ),
    AssetSpec(
        id="UI_Element_Wheel",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "minimalist mystical taoist five elements circular diagram emblem in "
            "black ink brush strokes with five glowing colored elemental orbs "
            "(gold, green, blue, red, amber) arranged in a circle, isolated on pure "
            "white background, 2d game UI icon"
        ),
        target_size=(512, 512),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=16,
        max_debris_ratio=0.30,
        padding_ratio=0.05,
    ),
    AssetSpec(
        id="Cursor_Brush",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "minimalist ancient chinese calligraphy brush tip pointing top-left with "
            "glowing cyan ink energy tip, black ink style, isolated on pure white "
            "background, 2d game cursor icon"
        ),
        target_size=(64, 64),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_fragments=10,
        max_debris_ratio=0.20,
        padding_ratio=0.06,
    ),
    AssetSpec(
        id="App_Icon",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "epic 2d game app icon, glowing cyan spiritual flying sword piercing "
            "through swirling black sumi-e ink clouds and golden Bagua runes on dark "
            "charcoal background, high contrast, no text"
        ),
        target_size=(512, 512),
        remove_bg=False,
        gen_size=(1024, 1024),
        mode="opaque_center",
        padding_ratio=0.0,
    ),
    AssetSpec(
        id="VFX_PhapTuong_Halo",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "mystical giant golden taoist celestial dharma wheel and glowing lotus "
            "aura ring, radiant golden calligraphy runes, sumi-e ink and gold style, "
            "isolated on pure white background, 2d game vfx sprite, centered"
        ),
        target_size=(512, 512),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=16,
        max_debris_ratio=0.25,
        padding_ratio=0.05,
    ),
    AssetSpec(
        id="Event_TanHon",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "ancient ghostly taoist spirit soul floating with glowing cyan and black "
            "ink smoke, mystical eastern fantasy illustration, sumi-e ink painting "
            "style, isolated on pure white background, centered"
        ),
        target_size=(256, 256),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=14,
        max_debris_ratio=0.25,
        padding_ratio=0.06,
    ),
    AssetSpec(
        id="Event_LoDan",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "ancient bronze chinese alchemy furnace cauldron overflowing with glowing "
            "golden and crimson spiritual fire, sumi-e ink painting style, isolated "
            "on pure white background, centered"
        ),
        target_size=(256, 256),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=14,
        max_debris_ratio=0.25,
        padding_ratio=0.06,
    ),
    AssetSpec(
        id="UI_Icon_Lock",
        category="UI_Kit",
        subfolder="UI_Kit",
        prompt=(
            "minimalist ancient chinese bronze padlock icon with glowing golden "
            "talisman seal, black ink outline, sumi-e style, isolated on pure white "
            "background, 2d game UI icon, centered"
        ),
        target_size=(128, 128),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=10,
        max_debris_ratio=0.20,
        padding_ratio=0.06,
    ),
    AssetSpec(
        id="VFX_Brush_Splash",
        category="Pickups",
        subfolder="Icons",
        prompt=(
            "single dynamic chinese calligraphy ink splatter blotch with sharp brush "
            "bristles and droplets, solid black ink on pure white background, "
            "minimalist sumi-e style, 2d game particle texture, centered"
        ),
        target_size=(128, 128),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=18,
        max_debris_ratio=0.28,
        padding_ratio=0.06,
    ),
    AssetSpec(
        id="VFX_Fire_Wisp",
        category="Pickups",
        subfolder="Icons",
        prompt=(
            "single stylized oriental spirit flame wisp curling upwards, black ink "
            "outline with glowing white-cyan inner core, sumi-e ink style, isolated "
            "on pure white background, 2d game particle texture, centered"
        ),
        target_size=(128, 128),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=12,
        max_debris_ratio=0.22,
        padding_ratio=0.06,
    ),
    AssetSpec(
        id="VFX_Ice_Shard",
        category="Pickups",
        subfolder="Icons",
        prompt=(
            "single sharp crystalline ice shard fragment with jagged edges and black "
            "ink sumi-e contour, glowing white-blue core, isolated on pure white "
            "background, 2d game particle texture, centered"
        ),
        target_size=(128, 128),
        remove_bg=True,
        gen_size=(1024, 1024),
        mode="icon_center",
        min_area_ratio=0.10,
        max_area_ratio=0.88,
        max_fragments=12,
        max_debris_ratio=0.22,
        padding_ratio=0.06,
    ),
)

ASSETS_BY_ID: dict[str, AssetSpec] = {spec.id: spec for spec in ASSETS}
