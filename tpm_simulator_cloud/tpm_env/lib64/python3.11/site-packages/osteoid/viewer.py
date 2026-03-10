import numpy as np

def view_matplotlib(
  skel, units='nm', 
  draw_edges=True, draw_vertices=True,
  color_by='radius'
):
    """
    View the skeleton with a radius heatmap. 

    Requires the matplotlib library which is 
    not installed by default.

    units: label axes with these units
    draw_edges: draw lines between vertices (more useful when skeleton is sparse)
    draw_vertices: draw each vertex colored by its radius.
    color_by: 
      'radius': color each vertex according to its radius attribute
        aliases: 'r', 'radius', 'radii'
      'component': color connected components seperately
        aliases: 'c', 'component', 'components'
      'cross_section': color each vertex according to its cross sectional area
        aliases: 'x'
      anything else: draw everything black
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    RADII_KEYWORDS = ('radius', 'radii', 'r')
    CROSS_SECTION_KEYWORDS = ('cross_section', 'x')
    COMPONENT_KEYWORDS = ('component', 'components', 'c')

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(units)
    ax.set_ylabel(units)
    ax.set_zlabel(units)

    # Set plot axes equal. Matplotlib doesn't have an easier way to
    # do this for 3d plots.
    X = skel.vertices[:,0]
    Y = skel.vertices[:,1]
    Z = skel.vertices[:,2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ### END EQUALIZATION CODE ###

    component_colors = ['k', 'deeppink', 'dodgerblue', 'mediumaquamarine', 'gold' ]

    def draw_component(i, skel):
      nonlocal units
      component_color = component_colors[ i % len(component_colors) ]

      if draw_vertices:
        xs = skel.vertices[:,0]
        ys = skel.vertices[:,1]
        zs = skel.vertices[:,2]

        if color_by in RADII_KEYWORDS or color_by in CROSS_SECTION_KEYWORDS:
          colmap = cm.ScalarMappable(cmap=cm.get_cmap('rainbow'))

          axis_label = ''
          if color_by in RADII_KEYWORDS:
            axis_label = 'radius'
            colmap.set_array(skel.radii)
            normed_data = skel.radii / np.max(skel.radii)
          else:
            axis_label = 'cross sectional area'
            units += '^2'
            colmap.set_array(skel.cross_sectional_area)
            normed_data = skel.cross_sectional_area / np.max(skel.cross_sectional_area)

          yg = ax.scatter(xs, ys, zs, c=cm.rainbow(normed_data), marker='o')
          cbar = fig.colorbar(colmap, ax=ax)
          cbar.set_label(f'{axis_label} ({units})', rotation=270)
        elif color_by in COMPONENT_KEYWORDS:
          yg = ax.scatter(xs, ys, zs, color=component_color, marker='.')
        else:
          yg = ax.scatter(xs, ys, zs, color='k', marker='.')

      if draw_edges:
        for e1, e2 in skel.edges:
          pt1, pt2 = skel.vertices[e1], skel.vertices[e2]
          ax.plot(  
            [ pt1[0], pt2[0] ],
            [ pt1[1], pt2[1] ],
            zs=[ pt1[2], pt2[2] ],
            color=(component_color if not draw_vertices else 'silver'),
            linewidth=1,
          )

    if color_by in COMPONENT_KEYWORDS:
      for i, skel in enumerate(skel.components()):
        draw_component(i, skel)
    else:
      draw_component(0, skel)

    plt.show()
