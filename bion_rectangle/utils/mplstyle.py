kw_marker_data = {
    'marker': 'o',
    'mfc': 'k',
    'mec': 'w',
    'linestyle': 'None',
}
kw_line = {
    'color': 'k',
    'linestyle': '-'
}
kw_trajectory = {
    **kw_line,
    'lw': 0.5
}
kw_plot = {
    'data': kw_marker_data,
    'pred': kw_line
}

color_deemed_correct = ['tab:blue', 'tab:orange', 'k']

def get_kw_plot(
        mode='data',
        deemed_correct=None
):
    color = (
        color_deemed_correct[2] if deemed_correct is None else
        color_deemed_correct[int(deemed_correct)]
    )
    if mode == 'data':
        return {**kw_marker_data, 'mfc': color}
    elif mode == 'pred':
        return {**kw_line, 'color': color}
    else:
        raise ValueError()

kw_errorbar = {
    'ecolor': 'k',
    'elinewidth': 0.5,
    'mew': 0.5,
}
kw_errorbar_data = {
    **kw_marker_data,
    **kw_errorbar
}
kw_axline = {
    'linestyle': '--',
    'lw': 0.5,
    'color': 'darkgray',
    'zorder': -1,
}
kw_bar = {
    # 'color': 'k',
    'facecolor': 'None',
    'edgecolor': 'k'
}
kw_legend = {
    'frameon': False,
    'handlelength': 0.8,
    'labelspacing': 0.5,
}
kw_legend_rightoutside = {
    **kw_legend,
    'loc': 'center left',
    'bbox_to_anchor': (1, 0, 1, 1),
}
kw_legend_leftoutside = {
    **kw_legend,
    'loc': 'center right',
    'bbox_to_anchor': (0, 0, 0, 1),
}
kw_legend_upperoutside = {
    **kw_legend,
    'loc': 'lower center',
    'bbox_to_anchor': (0.5, 1),
}

kw_legend_upperrightoutside = {
    **kw_legend,
    'loc': 'upper left',
    'bbox_to_anchor': (1, 0, 1, 1),
}

