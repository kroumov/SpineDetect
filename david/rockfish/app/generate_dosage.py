import numpy as np
from scipy.stats import poisson

from get_all_cts import get_all_cts


def generate_dosage(client, origin, serotypes, sigma, gc, p, seed=632, **kwargs):
    cts = get_all_cts(client)
    cts['soma_r'] = np.power(cts['volume'] * 3/(4*np.pi), 1/3)
    cts['soma_sa'] = np.power(cts['soma_r'], 2) * 4 * np.pi

    origin = np.array([origin['x'], origin['y'], origin['z']])
    KD = gc / (sigma * np.sqrt(2 * np.pi))

    rng = np.random.default_rng(seed)
    dose = dict()

    for row in cts.itertuples(index=False):
        if row.cell_type not in serotypes: continue

        neuron_pos = np.array(row.pt_position)
        rr = np.linalg.norm(origin - neuron_pos) / 1e3

        local_gc = row.soma_sa * KD * np.exp(-(np.pow(rr, 2))/(2*np.pow(sigma, 2)))
        moi = local_gc * p
        poi = poisson(moi)

        cell_dose = poi.rvs(1, random_state=rng)[0]
        if cell_dose != 0:
            dose[str(row.pt_root_id)] = int(cell_dose)

    return dose


if __name__ == '__main__':
    import json
    from caveclient import CAVEclient
    from pathlib import Path
    from filelock import FileLock

    filename = "dosage.json"

    project = Path('david/rockfish/example/')

    client = CAVEclient('minnie65_public')
    client.version = 1300

    volume = None
    with open(project.joinpath('volume.json'), 'r') as file:
        volume = json.load(file)

    args = volume['injection']

    dosage_path = project.joinpath(filename)
    if dosage_path.exists() != True:
        lock = FileLock(project.joinpath(filename + '.lock'))
        with lock:
            if dosage_path.exists() != True:
                dosage = generate_dosage(client, **args)
                with open(dosage_path, 'w') as file:
                    json.dump(dosage, file)
    