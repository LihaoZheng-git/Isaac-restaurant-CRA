"""Plated meals and bar prep for the parked restaurant preview.

The food is authored mesh dressing. It has no collision bodies and supplies
no perception, risk, or controller inputs. All edits go to the session layer.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from robot.scene.controlled_semantic_obstacle import read_world_bounds
LIVE_ROOT = "/World/RestaurantLiveV1"


FOOD_ROOT = "/World/RestaurantLiveDiningFood"
_RESTAURANT = "/World/Environment/Restaurant"
_LEFT = _RESTAURANT + "/assets_restaurant/Assets_Banquettes_Left02"
_FRONT = _RESTAURANT + "/assets_restaurant_Fixed/Assets_Banquettes_Front"
_RIGHT = _RESTAURANT + "/assets_restaurant/Assets_Banquettes_Right"
_BAR = _RESTAURANT + "/rh_Restaurant/Geometry/G_Arch/G_Bar/bar_ctop_low"
_TABLE_HEIGHT = .7466
_WELL_HEIGHT = .00350406


@dataclass(frozen=True)
class PlaceSetting:
    name: str
    table: str
    xy: tuple[float, float]
    facing: tuple[int, int]
    meal: str
    cup_side: int


SETTINGS = (
    PlaceSetting("LeftNearChair", _LEFT+"/Geo_TableSet_09/rh_table", (4.46, 17.00), (-1, 0), "sandwich", -1),
    PlaceSetting("LeftNearBooth", _LEFT+"/Geo_TableSet_09/rh_table", (4.00, 17.00), (1, 0), "salad", 1),
    PlaceSetting("LeftFarChair", _LEFT+"/Geo_TableSet_11/rh_table", (4.42, 20.25), (-1, 0), "pasta", 1),
    PlaceSetting("LeftFarBooth", _LEFT+"/Geo_TableSet_11/rh_table", (3.96, 20.25), (1, 0), "sandwich", -1),
    PlaceSetting("FrontChair", _FRONT+"/Geo_TableSet_08/rh_table", (13.05, 2.49), (0, -1), "salad", 1),
    PlaceSetting("FrontBooth", _FRONT+"/Geo_TableSet_08/rh_table", (13.05, 2.07), (0, 1), "pasta", -1),
    PlaceSetting("RightChair", _RIGHT+"/Geo_TableSet_14/rh_table", (31.23, 9.86), (1, 0), "pasta", -1),
    PlaceSetting("RightBooth", _RIGHT+"/Geo_TableSet_14/rh_table", (31.64, 9.86), (-1, 0), "salad", 1),
)


class _FoodMesh:
    def __init__(self, rng):
        self.rng = rng
        self.points = []
        self.colors = []
        self.faces = []

    def add(self, points, faces, color, variation=.09):
        start = len(self.points)
        points = np.asarray(points, dtype=float)
        colors = np.broadcast_to(np.asarray(color, dtype=float), points.shape)
        grain = self.rng.uniform(1-variation, 1+variation, (len(points), 1))
        self.points.extend(points.tolist())
        self.colors.extend(np.clip(colors*grain, 0, 1).tolist())
        self.faces.extend(tuple(start+i for i in face) for face in faces)

    def author(self, stage, path, material):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray(self.points))
        mesh.CreateFaceVertexCountsAttr([len(face) for face in self.faces])
        mesh.CreateFaceVertexIndicesAttr([index for face in self.faces for index in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        extent = np.asarray(self.points)
        mesh.CreateExtentAttr([Gf.Vec3f(*extent.min(axis=0)), Gf.Vec3f(*extent.max(axis=0))])
        UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
            Vt.Vec3fArray(self.colors))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _bands(ring_count, side_count):
    """Outward quads joining rings ordered from bottom to top."""
    return [(row*side_count+i, row*side_count+(i+1)%side_count,
             (row+1)*side_count+(i+1)%side_count, (row+1)*side_count+i)
            for row in range(ring_count-1) for i in range(side_count)]


def _ellipsoid(batch, center, radii, color, *, rough_edge=.04):
    rng = batch.rng
    sides = 12
    points = []
    for elev in np.linspace(-math.pi/2, math.pi/2, 8)[1:-1]:
        for theta in np.linspace(0, 2*math.pi, sides, endpoint=False):
            shape = np.array((math.cos(theta)*math.cos(elev), math.sin(theta)*math.cos(elev), math.sin(elev)))
            point = np.asarray(center) + radii*shape*rng.uniform(1-rough_edge, 1+rough_edge)
            points.append(point)
    faces = _bands(6, sides)
    bottom = len(points)
    points.extend([(center[0], center[1], center[2]-radii[2]), (center[0], center[1], center[2]+radii[2])])
    faces.extend((bottom, (i+1)%sides, i) for i in range(sides))
    faces.extend((bottom+1, 5*sides+i, 5*sides+(i+1)%sides) for i in range(sides))
    batch.add(points, faces, color)


def _leaf(batch, center, length, width, angle, color):
    """Curled leaf with a brighter midrib and irregular lobes."""
    rng = batch.rng
    x_axis = np.array((math.cos(angle), math.sin(angle)))
    y_axis = np.array((-x_axis[1], x_axis[0]))
    points, colors = [], []
    rows = 9
    for row, u in enumerate(np.linspace(0, 1, rows)):
        outline = max(.035, math.sin(math.pi*u))**.65
        for side in (-1, -.46, 0, .46, 1):
            scallop = 1 + .20*math.sin(row*2.8 + side) + rng.uniform(-.08, .08)
            xy = center[:2] + x_axis*(u-.5)*length + y_axis*(side*width*outline*scallop)
            z = center[2] + .0038*math.sin(math.pi*u) + abs(side)*.004*math.sin(row*1.6+side)
            points.append([*xy, max(.0003, z)])
            colors.append(np.asarray(color)*(1.42 if side == 0 else 1.05-.19*abs(side)))
    faces = [(r*5+c, (r+1)*5+c, (r+1)*5+c+1, r*5+c+1) for r in range(rows-1) for c in range(4)]
    batch.add(points, faces, colors, .15)


def _greens(batch, center, count, radius, *, lift=.0):
    rng = batch.rng
    for index in range(count):
        theta = rng.uniform(0, 2*math.pi)
        dist = radius*math.sqrt(rng.random())
        xy = np.asarray(center[:2]) + dist*np.array((math.cos(theta), math.sin(theta)))
        z = center[2] + lift + .0075*(1-dist/radius) + rng.uniform(0, .009)
        if index == 0:
            z = center[2] + lift
        color = (.095, .26, .029) if index%3 else (.033, .145, .019)
        _leaf(batch, (*xy, z), rng.uniform(.026, .054), rng.uniform(.006, .013), rng.uniform(0, 2*math.pi), color)


def _tomatoes(batch, center, positions):
    for x, y, z in positions:
        xyz = (center[0]+x, center[1]+y, center[2]+z)
        _ellipsoid(batch, xyz, (.0107, .0084, .0056), (.57, .034, .019))
        for theta in (0.25, 2.25, 4.35):
            seed = (xyz[0]+.0038*math.cos(theta), xyz[1]+.003*math.sin(theta), xyz[2]+.005)
            _ellipsoid(batch, seed, (.0017, .001, .00065), (.70, .35, .081))


def _bread(batch, center):
    rng = batch.rng
    sides = 40
    angles = np.linspace(0, 2*math.pi, sides, endpoint=False)
    xy = np.asarray([(math.copysign(abs(math.cos(a))**.49, math.cos(a))*.054,
                      math.copysign(abs(math.sin(a))**.57, math.sin(a))*.047) for a in angles])
    xy *= rng.uniform(.97, 1.03, (sides, 1))
    profile = ((.81, 0), (.97, .003), (1, .009), (.97, .015), (.86, .019), (.61, .020), (.33, .0204), (.10, .0208))
    points, colors = [], []
    for index, (radius, z) in enumerate(profile):
        for x, y in radius*xy:
            points.append((center[0]+x, center[1]+y, center[2]+z))
            toasted = 1 - .37*math.exp(-((x+.013)/.025)**2-((y-.011)/.027)**2)
            color = (.32, .101, .019) if index < 4 else (.64, .334, .108)
            colors.append(np.asarray(color)*toasted)
    faces = _bands(len(profile), sides)
    faces += [tuple(reversed(range(sides))), tuple(range((len(profile)-1)*sides, len(profile)*sides))]
    batch.add(points, faces, colors, .14)


def _noodles(batch):
    rng = batch.rng
    sides = 6
    for strand in range(52):
        radial_angle = rng.uniform(0, 2*math.pi)
        radial_dist = .048*math.sqrt(rng.random())
        offset = radial_dist*np.asarray((math.cos(radial_angle), math.sin(radial_angle)))
        base = .0017 if strand < 15 else rng.uniform(.006, .032)
        start = rng.uniform(0, 2*math.pi)
        angles = np.linspace(start, start+rng.uniform(2.9, 5.7), 22)
        rx, ry = rng.uniform(.015, .032, 2)
        points = []
        for theta in angles:
            xy = offset + np.array((rx*math.cos(theta), ry*math.sin(theta)))
            z = base + .0038*math.sin(theta*2.4 + strand)
            for around in np.linspace(0, 2*math.pi, sides, endpoint=False):
                r = .00165*math.cos(around)
                points.append((* (xy + r*np.array((math.cos(theta), math.sin(theta)))),
                               max(.0001, z+.00165*math.sin(around))))
        faces = _bands(len(angles), sides)
        faces += [tuple(reversed(range(sides))), tuple(range((len(angles)-1)*sides, len(angles)*sides))]
        color = rng.uniform(.83, 1.05)*np.array((.71, .46, .17))
        batch.add(points, faces, color, .06)


def _meal_mesh(meal, seed):
    rng = np.random.default_rng(seed)
    mesh = _FoodMesh(rng)
    if meal == "sandwich":
        _bread(mesh, (-.009, -.004, 0))
        _ellipsoid(mesh, (-.008, -.005, .024), (.049, .037, .008), (.39, .054, .026), rough_edge=.10)
        _greens(mesh, (-.013, -.004, .028), 21, .035)
        _bread(mesh, (-.003, -.007, .044))
        _greens(mesh, (.051, .044, .0), 8, .016)
        _tomatoes(mesh, (0, 0, 0), [(.061, -.026, .006), (.071, -.005, .008), (.053, .026, .013)])
    elif meal == "salad":
        _greens(mesh, (0, 0, 0), 47, .062)
        _tomatoes(mesh, (0, 0, 0), [(.054, .014, .018), (-.053, .034, .016), (.022, -.049, .018),
                                   (-.028, -.023, .026), (.030, .052, .015), (-.047, -.038, .016)])
        for i in range(14):
            xy = rng.uniform(-.051, .051, 2)
            _ellipsoid(mesh, (*xy, rng.uniform(.016, .025)), (.0041, .0032, .0025), (.70, .57, .32))
    elif meal == "pasta":
        _noodles(mesh)
        for i in range(14):
            xy = rng.uniform(-.034, .034, 2)
            _ellipsoid(mesh, (*xy, rng.uniform(.024, .033)), (.013, .010, .006), (.47, .040, .013), rough_edge=.22)
        for angle in (.1, 2.3, 4.8):
            _leaf(mesh, (0, 0, .041), .038, .009, angle, (.043, .25, .031))
    else:
        raise ValueError(f"Unknown plated meal {meal}")
    return mesh


def _food_material(stage):
    path = FOOD_ROOT+"/FoodMaterial"
    material = UsdShade.Material.Define(stage, path)
    preview = UsdShade.Shader.Define(stage, path+"/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.48)
    colors = UsdShade.Shader.Define(stage, path+"/Colors")
    colors.CreateIdAttr("UsdPrimvarReader_float3")
    colors.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
    colors.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(.35, .13, .04))
    colors.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(colors.ConnectableAPI(), "result")
    preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    return material


def _surface_triangles(stage, path, height):
    root = stage.GetPrimAtPath(path)
    if not root:
        raise ValueError(f"Missing support prim {path}")
    transforms = UsdGeom.XformCache()
    triangles = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = transforms.GetLocalToWorldTransform(prim)
        points = np.asarray([matrix.Transform(p) for p in mesh.GetPointsAttr().Get()])
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        offset = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = points[np.asarray(indices[offset:offset+count])]
            offset += count
            if count >= 3 and np.all(np.abs(face[:, 2]-height)<2.e-5):
                triangles.extend((face[0, :2], face[i, :2], face[i+1, :2]) for i in range(1, count-1))
    if not triangles:
        raise ValueError(f"No horizontal support triangles near z={height} at {path}")
    return np.asarray(triangles)


def _supported(triangles, xy):
    p0, p1, p2 = triangles.transpose(1, 0, 2)
    a, b, c = p1-p0, p2-p0, np.asarray(xy)-p0
    det = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    valid = np.abs(det)>1.e-12
    det = np.where(valid, det, 1)
    u = (c[:, 0]*b[:, 1]-c[:, 1]*b[:, 0])/det
    v = (a[:, 0]*c[:, 1]-a[:, 1]*c[:, 0])/det
    return bool(np.any(valid & (u>=-1.e-5) & (v>=-1.e-5) & (u+v<=1+1.e-5)))


def _place_asset(stage, path, asset_root, asset, position, triangles, *, angle=0., scale=1.):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.GetPrim().GetReferences().AddReference((asset_root/(asset+".usdc")).as_posix())
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateZOp().Set(angle)
    xform.AddScaleOp().Set(Gf.Vec3d(scale))
    bounds = read_world_bounds(stage, path)
    samples = [(float(x), float(y))
               for x in np.linspace(bounds.minimum[0], bounds.maximum[0], 3)
               for y in np.linspace(bounds.minimum[1], bounds.maximum[1], 3)]
    misses = [xy for xy in samples if not _supported(triangles, xy)]
    if misses or abs(bounds.minimum[2]-position[2])>1.e-5:
        raise ValueError(f"Unsupported tabletop footprint at {path}: {misses}")
    return dict(path=path, asset=asset, base_xyz=position, supported_samples=len(samples),
                bounds_minimum=bounds.minimum, bounds_maximum=bounds.maximum)


def _place_meal(stage, path, plate, meal, seed, material):
    x, y, z = plate["base_xyz"]
    height = z+_WELL_HEIGHT
    if not _supported(_surface_triangles(stage, plate["path"], height), (x, y)):
        raise ValueError(f"Plate well height could not be measured at {plate['path']}")
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(x, y, height))
    root.AddRotateZOp().Set(12.*(seed%5)-24.)
    if plate["asset"] == "side_plate":
        root.AddScaleOp().Set(Gf.Vec3d(.72))
    mesh = _meal_mesh(meal, seed)
    mesh.author(stage, path+"/Ingredients", material)
    bounds = read_world_bounds(stage, path)
    if any(bounds.minimum[i]<plate["bounds_minimum"][i] or bounds.maximum[i]>plate["bounds_maximum"][i]
           for i in range(2)) or not -1.e-6 <= bounds.minimum[2]-height <= .002:
        raise ValueError(f"Meal does not fit the plate well at {path}: {bounds}")
    return dict(path=path, meal=meal, plate=plate["path"], base_xyz=(x, y, height),
                bounds_minimum=bounds.minimum, bounds_maximum=bounds.maximum,
                plate_center_support_verified=True)


def author_dining_food(stage, *, asset_root):
    """Add eight plated meals and a prep plate, with world support receipts."""
    asset_root = Path(asset_root).resolve()
    for asset in ("dinner_plate", "side_plate", "fork", "latte_saucer"):
        if not (asset_root/(asset+".usdc")).is_file():
            raise FileNotFoundError(asset_root/(asset+".usdc"))
    if stage.GetPrimAtPath(FOOD_ROOT):
        raise ValueError("Dining food has already been authored")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        return _author_dining_food(stage, asset_root)


def _author_dining_food(stage, asset_root):
    UsdGeom.Xform.Define(stage, FOOD_ROOT)
    old_dressing = LIVE_ROOT+"/Dressing/"
    replaced = []
    old_names = ["MiddleLatte", "MiddleSidePlate", "FrontDinnerPlate", "FrontLatte", "FrontSmallBowl"]
    old_names.extend("Middle"+place+item for place in "AB" for item in ("Plate", "Fork", "Knife"))
    for name in old_names:
        prim = stage.GetPrimAtPath(old_dressing+name)
        if prim and prim.IsActive():
            prim.SetActive(False)
            replaced.append(str(prim.GetPath()))
    material = _food_material(stage)
    surfaces = {table: _surface_triangles(stage, table, _TABLE_HEIGHT) for table in {s.table for s in SETTINGS}}
    records = []
    for index, setting in enumerate(SETTINGS):
        path = FOOD_ROOT+"/"+setting.name
        UsdGeom.Xform.Define(stage, path)
        x, y = setting.xy
        side_x, side_y = -setting.facing[1]*setting.cup_side, setting.facing[0]*setting.cup_side
        triangles = surfaces[setting.table]
        plate = _place_asset(stage, path+"/Plate", asset_root, "dinner_plate", (x, y, _TABLE_HEIGHT), triangles)
        meal = _place_meal(stage, path+"/Meal", plate, setting.meal, index+30, material)
        fork_angle = 90. if setting.facing[0] else 0.
        fork = _place_asset(stage, path+"/Fork", asset_root, "fork", (x-.174*side_x, y-.174*side_y, _TABLE_HEIGHT),
                            triangles, angle=fork_angle, scale=.72)
        cup = _place_asset(stage, path+"/Coffee", asset_root, "latte_saucer", (x+.231*side_x, y+.231*side_y, _TABLE_HEIGHT),
                           triangles, angle=35.+index*37.)
        records.append(dict(name=setting.name, tabletop=setting.table, plate=plate, meal=meal, fork=fork, coffee=cup,
                            gesture_target_xyz=(x, y, meal["bounds_maximum"][2]+.027)))
    bar_plate = _place_asset(stage, FOOD_ROOT+"/BarPrepPlate", asset_root, "side_plate", (23.103, 6.294, .90),
                             _surface_triangles(stage, _BAR, .90))
    prep = _place_meal(stage, FOOD_ROOT+"/BarPrepMeal", bar_plate, "sandwich", 48, material)
    return dict(settings=records, bar_prep=dict(plate=bar_plate, meal=prep, gesture_target_xyz=(23.09, 6.34, .95)),
                previous_settings_hidden=replaced, food_source="Procedural plated food geometry authored in this scene",
                dishware_source="Existing exported Inge Dinnerware and kitchen utensil assets")


# Additional paired and family tables from the recorded restaurant setup.
_BANKS = "/World/Environment/Restaurant/assets_restaurant"
_LEFT_TABLE = _BANKS + "/Assets_Banquettes_Left02/Geo_TableSet_14/rh_table"
_RIGHT_TABLE = _BANKS + "/Assets_Banquettes_Right/Geo_TableSet_11/rh_table"
_HEIGHT = .7466

ADDITIONAL_SETTINGS = (
    PlaceSetting("LeftOppositeChair", _LEFT_TABLE, (1.94, 19.51), (1, 0), "salad", 1),
    PlaceSetting("LeftOppositeBooth", _LEFT_TABLE, (2.36, 19.51), (-1, 0), "pasta", 1),
    PlaceSetting("RightOppositeChair", _RIGHT_TABLE, (33.71, 9.03), (-1, 0), "sandwich", -1),
    PlaceSetting("RightOppositeBooth", _RIGHT_TABLE, (33.25, 9.03), (1, 0), "salad", -1),
)


def author_occupancy_food(stage, *, asset_root):
    """Extend the plated menu with four supported place settings."""
    asset_root = Path(asset_root).resolve()
    food = author_dining_food(stage, asset_root=asset_root)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        material = _food_material(stage)
        surfaces = {table: _surface_triangles(stage, table, _HEIGHT)
                    for table in {_LEFT_TABLE, _RIGHT_TABLE}}
        for index, setting in enumerate(ADDITIONAL_SETTINGS):
            path = FOOD_ROOT + "/" + setting.name
            UsdGeom.Xform.Define(stage, path)
            x, y = setting.xy
            side = setting.facing[0] * setting.cup_side
            triangles = surfaces[setting.table]
            plate = _place_asset(stage, path + "/Plate", asset_root, "dinner_plate",
                                 (x, y, _HEIGHT), triangles)
            meal = _place_meal(stage, path + "/Meal", plate, setting.meal, 60 + index, material)
            fork = _place_asset(stage, path + "/Fork", asset_root, "fork",
                                (x, y - .174 * side, _HEIGHT), triangles, angle=90., scale=.72)
            cup = _place_asset(stage, path + "/Coffee", asset_root, "latte_saucer",
                               (x, y + .231 * side, _HEIGHT), triangles, angle=43. + index * 37.)
            food["settings"].append(dict(name=setting.name, tabletop=setting.table,
                                         plate=plate, meal=meal, fork=fork, coffee=cup,
                                         gesture_target_xyz=(x, y, meal["bounds_maximum"][2] + .027)))
    return food


_FAMILY_TABLE = _BANKS + "/Assets_Banquettes_Right_01/Geo_TableSet_14/rh_table"
_SOLO_TABLE = _BANKS + "/Assets_Banquettes_Right_Wall/Geo_TableSet_11/rh_table"

FAMILY_CHILD_MEAL_TARGET = (31.23, 19.51042, .82)
_FAMILY_SETTINGS = (
    PlaceSetting("FamilyFather", _FAMILY_TABLE, (31.23, 20.29285), (1, 0), "pasta", -1),
    PlaceSetting("FamilyMother", _FAMILY_TABLE, (31.66, 19.90), (-1, 0), "salad", -1),
    PlaceSetting("FamilyChild", _FAMILY_TABLE, FAMILY_CHILD_MEAL_TARGET[:2], (1, 0), "sandwich", 1),
    PlaceSetting("SoloDiner", _SOLO_TABLE, (28.298837, 19.462805), (-1, 0), "sandwich", -1),
)


def author_family_food(stage, *, asset_root):
    """Add the family and solo place settings to the supported plated menu."""
    asset_root = Path(asset_root).resolve()
    food = author_occupancy_food(stage, asset_root=asset_root)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        material = _food_material(stage)
        surfaces = {path: _surface_triangles(stage, path, _HEIGHT) for path in (_FAMILY_TABLE, _SOLO_TABLE)}
        for index, setting in enumerate(_FAMILY_SETTINGS):
            path = FOOD_ROOT + "/" + setting.name
            UsdGeom.Xform.Define(stage, path)
            x, y = setting.xy
            side = setting.facing[0] * setting.cup_side
            triangles = surfaces[setting.table]
            dish = "side_plate" if setting.name == "FamilyChild" else "dinner_plate"
            plate = _place_asset(stage, path + "/Plate", asset_root, dish, (x, y, _HEIGHT), triangles)
            meal = _place_meal(stage, path + "/Meal", plate, setting.meal, 70 + index, material)
            fork = _place_asset(stage, path + "/Fork", asset_root, "fork",
                                (x, y - .174 * side, _HEIGHT), triangles, angle=90., scale=.72)
            cup = _place_asset(stage, path + "/Coffee", asset_root, "latte_saucer",
                               (x, y + .24 * side, _HEIGHT), triangles, angle=43. + index * 37.)
            food["settings"].append(dict(name=setting.name, tabletop=setting.table,
                                         plate=plate, meal=meal, fork=fork, coffee=cup,
                                         gesture_target_xyz=(x, y, meal["bounds_maximum"][2] + .027)))
    return food
