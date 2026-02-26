import pandas as pd

def get_all_cts(client):
    ct_manual_df = client.materialize.tables \
        .allen_v1_column_types_slanted_ref() \
        .query(desired_resolution=[1,1,1]) \
        .rename(columns={'target_id': 'nucleus_id'}) \
        .drop_duplicates('pt_root_id', keep=False)

    ct_auto_df = client.materialize.tables \
        .aibs_metamodel_celltypes_v661() \
        .query(desired_resolution=[1,1,1]) \
        .rename(columns={'target_id': 'nucleus_id'}) \
        .drop_duplicates('pt_root_id', keep=False)

    indices = ['pt_root_id', 'classification_system', 'cell_type', 'nucleus_id', 'pt_position', 'volume']

    ct_all_df = pd.merge(
        ct_auto_df[indices], ct_manual_df[indices], 
        on=['pt_root_id', 'nucleus_id'], 
        how='outer', 
        suffixes=['_auto', '_manual']
        )

    merged_cols = ['cell_type', 'classification_system', 'pt_position']
    for col in [*merged_cols, 'volume']:
        ct_all_df[col] = ct_all_df[f'{ col }_manual'].fillna(ct_all_df[f'{ col }_auto'])

    ct_all_df = ct_all_df.fillna({
        f'{ col }_auto': 'unknown'    
        for col in merged_cols
    })

    ct_all_df = ct_all_df.fillna({
        f'{ col }_manual': 'unknown'
        for col in merged_cols
    })

    return ct_all_df


if __name__ == '__main__':
    from caveclient import CAVEclient

    client = CAVEclient('minnie65_public')
    client.version = 1300

    cts = get_all_cts(client)
    cts.head().to_csv('cts.csv')