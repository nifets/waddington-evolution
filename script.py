import numpy as np
import bpy
import colorsys

import random
import sys
from scipy.stats.qmc import PoissonDisk

from abc import ABC, abstractmethod
import bmesh

NUM_FRAMES = 200

LANDSCAPE_Z = 14.0
GENE_PEG_MESH = bpy.data.objects['gene_prototype'].copy()
SPHERE_MESH = bpy.data.objects['deformer_sphere'].copy()

COLLECTION = bpy.data.collections.new("WaddingtonCollection")
bpy.context.scene.collection.children.link(COLLECTION)

def get_frame(t):
    return int(t* NUM_FRAMES)


class PlaneSampler:
    def __init__(self, side_length):
        self.side_length = side_length
#        self.poisson_disk = PoissonDisk(d=2, radius=poisson_disk_radius, 
#                                        l_bounds=(-gene_square_side/2,-gene_square_side/2),
#                                        u_bounds=(gene_square_side/2,gene_square_side/2))     
        
    def random(self, n):
        return np.random.uniform(-self.side_length/2, self.side_length/2, (n, 2))

# material stuff
class MaterialSelector:
    def __init__(self, num_colours=32, s=1.0, v=1.0):
        self.num_colours = num_colours
        self.materials = []
        for i in range(num_colours):
            hue = i / num_colours  # Evenly distribute hue values
            rgb = colorsys.hsv_to_rgb(hue, s, v)
            mat_name = f"GeneMaterial_{hue:.3f}"
            if mat_name not in bpy.data.materials:
                material = bpy.data.materials.new(mat_name)
            else:
                material =  bpy.data.materials[mat_name]
            material.use_nodes = True
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (*rgb, 1)
            
            self.materials.append(material)
            
    def get(self, hue):
        index = int((hue % 1.0) * self.num_colours)
        return self.materials[index]
        
    
MATERIAL_SELECTOR = MaterialSelector(num_colours=100, s=0.6, v=0.5)



#class Landscape:
#    def __init__(self):
#        mesh = bpy.data.meshes.new("ClothMesh")
#        self.obj = bpy.data.objects.new("Cloth", mesh)
#        COLLECTION.objects.link(self.obj)
#        
#        bm = bmesh.new()
#        bmesh.ops.create_grid(bm, x_segments=20, y_segments=20, size = 20)
#        
#        bm.to_mesh(mesh)
#        bm.free()
#        
#        self.cloth_mod = self.obj.modifiers.new(name="ClothSim", type='CLOTH')
#        
#        bpy.app.handlers.frame_change_pre.clear()
#        bpy.app.handlers.frame_change_pre.append(self.update)
#        
#    def update(self, scene):
#        frame = scene.frame_current
#        self.obj.location = (0, 0, 15)
#        self.cloth_mod.point_cache.frame_start = frame
#        
#        cache = self.cloth_mod.point_cache
#        cache.frame_start = frame
#        cache.frame_end = frame + 30
#        cache.reset()
    
class AnimatedMesh:
    def __init__(self, obj, hide_initially=True):
        self.obj = obj
        COLLECTION.objects.link(self.obj)
        
        self._set_hide(0, hide_initially)
        
    def appear(self, frame):
        self._set_hide(frame, False)
    
    def disappear(self, frame):
        self._set_hide(frame, True)
        
    def _set_hide(self, frame, setting):
        self.obj.hide_viewport = setting
        self.obj.hide_render = setting
        self.obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        self.obj.keyframe_insert(data_path="hide_render", frame=frame)
    
    def move(self, loc, frame):
        self.obj.location = loc
        self.obj.keyframe_insert(data_path="location", frame=frame)
    
    
class GeneMesh(AnimatedMesh):
    def __init__(self, loc_x, loc_y, loc_z=GENE_PEG_MESH.location[2], hue=0.0):
        obj = GENE_PEG_MESH.copy()
        obj.data = GENE_PEG_MESH.data.copy()
        
        obj.location = (loc_x, loc_y, loc_z)
        
        material = MATERIAL_SELECTOR.get(hue)
        obj.data.materials[0] = material
        
        super().__init__(obj)  
    
    
class RopeMesh(AnimatedMesh):
    def __init__(self, loc0, loc1, res_u=6, bevel_depth=0.05, bevel_res=5):
        data = bpy.data.curves.new(name="Curve", type="CURVE")
        data.dimensions = '3D'
        data.resolution_u = res_u # resolution across curve length
        polyline = data.splines.new('BEZIER')
        polyline.bezier_points.add(1)
        
        p0 = polyline.bezier_points[0]
        p0.co = loc0
        p0.handle_left_type = p0.handle_right_type = 'AUTO'
        p1 = polyline.bezier_points[1]
        p1.co = loc1
        p1.handle_left_type = p1.handle_right_type = 'AUTO'
        
        data.bevel_depth = bevel_depth
        data.bevel_resolution = bevel_res
        
        
        self.p0 = p0
        self.p1 = p1
        obj = bpy.data.objects.new("CurveObj", data)
        
        self.p0.keyframe_insert(data_path="co", frame=0)
        self.p1.keyframe_insert(data_path="co", frame=0)
        
        obj.data.materials.clear()
        obj.data.materials.append(bpy.data.materials["BlackMaterial"])
        
        super().__init__(obj)
        
    def move_root(self, loc, frame):
        self.p0.co = loc
        self.p0.keyframe_insert(data_path="co", frame=frame)
        
    def move_tip(self, loc, frame):
        self.p1.co = loc
        self.p1.keyframe_insert(data_path="co", frame=frame)
        
class DeformerMesh(AnimatedMesh):
    def __init__(self, loc, scale=1.0):
        obj = SPHERE_MESH.copy()
        obj.data = SPHERE_MESH.data.copy()
        
        #obj.modifiers.new(name="Collision", type='COLLISSION')
        
        obj.location = loc
        
        obj.scale = (scale, scale, scale)
        
        super().__init__(obj, hide_initially=False)  

class Deformer:
    def __init__(self, loc_xy, node_loc, scale):
        self.loc = (loc_xy[0], loc_xy[1], LANDSCAPE_Z)
        self.sphere_mesh = DeformerMesh(self.loc, scale=scale)
        self.rope_mesh = RopeMesh(loc0=node_loc, loc1=self.loc, bevel_depth=0.06)
        self.rope_mesh.appear(0)
        self.rope_mesh.move_root(node_loc, 0)
        self.rope_mesh.move_tip(self.loc, 0)
    
    def update(self, node_loc, frame):
        # jiggle around
        self.loc = (self.loc[0] + random.uniform(-0.3, 0.3),
                    self.loc[1] + random.uniform(-0.3, 0.3),
                    self.loc[2])
        self.rope_mesh.move_root(node_loc, frame)
        self.rope_mesh.move_tip(self.loc, frame)
        self.sphere_mesh.move(self.loc, frame)
        
    def appear(self, frame):
        self.rope_mesh.appear(frame)
        self.sphere_mesh.appear(frame)
        
    def disappear(self, frame):
        self.rope_mesh.disappear(frame)
        self.sphere_mesh.disappear(frame)
                    
        
        

class Node:
    def __init__(self, loc, sampler, num_deformers=3):
        self.loc = loc
        self.sampler = sampler
        locs = sampler.random(num_deformers)
        scales = np.random.uniform(0.4, 1.9, num_deformers)
        self.deformers = [Deformer(loc, self.loc, scale) for loc, scale in zip(locs, scales)]
        
        self.genes = []
        
    def update(self, frame):
        if not self.genes:
            for deformer in self.deformers:
                deformer.disappear(frame)
        else:
            for deformer in self.deformers:
                deformer.appear(frame)
        
        self.loc = (self.loc[0] + random.uniform(-0.2, 0.2),
                    self.loc[1] + random.uniform(-0.2, 0.2),
                    self.loc[2])
        for deformer in self.deformers:
            deformer.update(self.loc, frame)
            
    def add_gene(self, gene):
        self.genes.append(gene)
        
    def remove_gene(self, gene):
        self.genes.remove(gene)
    

class Gene:
    def __init__(self, birth_frame, hue, location, node):
        
        # set up parent node
        self.node = node
        
        # set up meshes
        self.peg_mesh = GeneMesh(loc_x=location[0], loc_y=location[1], hue=hue)
        
        rope_loc_base = (location[0], 
                         location[1], 
                         self.peg_mesh.obj.location.z+self.peg_mesh.obj.dimensions.z / 2)
        rope_loc_tip = self.node.loc
        self.rope_loc_base = rope_loc_base
        self.rope_mesh = RopeMesh(loc0=rope_loc_base, 
                                  loc1=rope_loc_tip,
                                  bevel_depth=0.03)
        
        self.peg_mesh.appear(birth_frame)
        self.rope_mesh.appear(birth_frame)
        self.rope_mesh.move_root(rope_loc_base, birth_frame)
        self.rope_mesh.move_tip(self.node.loc, birth_frame)
        
        self.node.add_gene(self)
        
        

    
    def kill(self, death_frame):
        self.node.remove_gene(self)
        self.peg_mesh.disappear(death_frame)
        self.rope_mesh.disappear(death_frame)
        
    
    def update(self, frame, new_node=None):
        if new_node is not None:
            self.node.remove_gene(self)
            self.node=new_node
            self.node.add_gene(self)
            
        self.rope_mesh.move_root(self.rope_loc_base, frame)
        self.rope_mesh.move_tip(self.node.loc, frame)
        
        

    

class GRN:
    
    def __init__(self, 
                 num_initial_genes=20,
                 num_initial_nodes=10,
                 initial_time=0.05,
                 gene_square_side=20.0,
                 poisson_disk_radius=6.0,
                 death_rate=1.0,
                 rewire_rate=2.0):
        self.curr_time = initial_time              
                     
        # set up sampler for gene locations and node locations
        self.gene_loc_sampler = PlaneSampler(gene_square_side)
        self.node_loc_sampler = PlaneSampler(gene_square_side * 0.75)
              
        # generate initial nodes  
        node_locs = self.node_loc_sampler.random(num_initial_nodes)
        self.nodes = [Node((loc[0],loc[1],3.0), sampler=self.gene_loc_sampler) for loc in node_locs]
        
        
        # generate initial genes
        birth_times = np.random.uniform(low=0.0, high=initial_time, size=num_initial_genes)
        gene_locs = self.gene_loc_sampler.random(num_initial_genes)
        gene_nodes = random.choices(self.nodes, k=num_initial_genes)
        
        self.genes = [Gene(get_frame(t), hue=t, location=loc, node=node) 
                      for t,loc,node in zip(birth_times, gene_locs, gene_nodes)]
        

        
        self.death_rate=death_rate
        self.rewire_rate=rewire_rate
        
                                        

        
        
    def simulate(self, num_steps, delta_time=0.03):
        for _ in range(num_steps):
            self.update(delta_time=delta_time)
        
        
    def update(self, delta_time):
        self.curr_time += delta_time
        frame = get_frame(self.curr_time)
        
        # kill some genes
        p = min(1, delta_time * self.death_rate)
        is_deads = np.random.binomial(1, p, size=len(self.genes))

        for gene, is_dead in zip(self.genes, is_deads):
            if is_dead:
                gene.kill(frame)
        
        self.genes = [gene for gene, is_dead in zip(self.genes, is_deads) if not is_dead]
        
        # create some genes
        
        loc = self.gene_loc_sampler.random(1)
        if len(loc) > 0:
            node = random.choices(self.nodes, k=1)[0]
            new_gene = Gene(frame, hue=self.curr_time, location=loc[0], node=node)
            self.genes.append(new_gene)
            
            
        # update nodes
        for node in self.nodes:
            node.update(frame)
            
        # rewire some genes
        p = min(1, delta_time * self.rewire_rate)
        is_rewireds = np.random.binomial(1, p, size=len(self.genes))
        
        new_nodes = []
        for gene, is_rewired in zip(self.genes, is_rewireds):
            if is_rewired:
                new_nodes.append(random.choices(self.nodes, k=1)[0])
            else:
                new_nodes.append(None)
                
            
        # update genes
        for gene, node in zip(self.genes, new_nodes):
            gene.update(frame, new_node=node)
        
        
        
        
        
        # reorganise the network, jiggle nodes around
        
        # optionally add a new node and shift some genes to it
        
        
        # sample how many new genes to spawn and create them
        
        # change the parents of some genes at random
        

my_grn = GRN(num_initial_genes=10, poisson_disk_radius=0.5)

my_grn.simulate(num_steps=200)
        