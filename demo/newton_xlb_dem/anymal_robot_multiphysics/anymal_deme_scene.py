"""DEME particle and foot-proxy construction for the ANYmal experiment."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AnymalDEMEPopulation:
    sampler: object
    sampled_positions: object
    positions_numpy: np.ndarray
    radius_indices: np.ndarray
    radii_numpy: np.ndarray


@dataclass
class AnymalDEMEScene:
    solver: object
    material: object
    ground_boundary: object
    shank_templates: list
    shanks: list
    shank_trackers: list
    particle_templates: list
    used_particle_templates: list
    particles: object
    particles_tracker: object


def prepare_anymal_deme_population(
    deme_module,
    radius_types,
    sample_center,
    sample_half_dimensions,
    random_seed,
):
    """Sample deterministic particle positions and radius assignments."""
    sampler = deme_module.PDSampler(2.0 * max(radius_types))
    sampled_positions = sampler.SampleBox(sample_center, sample_half_dimensions)
    positions_numpy = np.array(sampled_positions, dtype=np.float32)
    generator = np.random.default_rng(seed=random_seed)
    radius_indices = generator.integers(0, len(radius_types), size=len(sampled_positions))
    radii_numpy = np.array([radius_types[index] for index in radius_indices], dtype=np.float32)
    return AnymalDEMEPopulation(sampler, sampled_positions, positions_numpy, radius_indices, radii_numpy)


def build_anymal_deme_scene(
    deme_module,
    foot_radii,
    radius_types,
    population,
    material_properties,
    gravity,
    average_contact_limit,
    fixed_family,
    initial_particle_velocity,
    timestep,
):
    """Build and initialize DEME foot proxies, particles, and their trackers."""
    solver = deme_module.DEMSolver()
    material = solver.LoadMaterial(material_properties)
    ground_boundary = solver.AddBCPlane([0, 0, 0], [0, 0, 1], material)
    solver.SetGravitationalAcceleration(gravity)
    solver.SetErrorOutAvgContacts(average_contact_limit)
    shank_templates, shanks, shank_trackers = [], [], []
    for radius in foot_radii:
        template = solver.LoadSphereType(1.0, radius, material)
        shank = solver.AddClumps(template, [[0.0, 0.0, 0.0]])
        shank.SetFamily(fixed_family)
        shank_templates.append(template)
        shanks.append(shank)
        shank_trackers.append(solver.Track(shank))
    solver.SetFamilyFixed(fixed_family)
    particle_templates = [solver.LoadSphereType(1.0, radius, material) for radius in radius_types]
    used_particle_templates = [particle_templates[index] for index in population.radius_indices]
    particles = solver.AddClumps(used_particle_templates, population.positions_numpy)
    particles.SetVel(initial_particle_velocity)
    particles_tracker = solver.Track(particles)
    solver.DisableAdaptiveBinSize()
    solver.SetInitTimeStep(timestep)
    solver.Initialize()
    return AnymalDEMEScene(
        solver,
        material,
        ground_boundary,
        shank_templates,
        shanks,
        shank_trackers,
        particle_templates,
        used_particle_templates,
        particles,
        particles_tracker,
    )
