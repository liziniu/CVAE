def plot_marg(cen_lst, cov_lst, p_k, X, draw_scatter=False, draw_ellipses=False, \
        draw_pcolor=True, scatter_color='k', ellipse_color='k', fig=None, vmin=None, vmax=None, \
        labels=None, title=''):
    """
    """
    fig = figure()
    D = cen_lst[0].shape[0]
    plot_width = 1.0 / D
    left_offset = plot_width/2
    bottom_offset = plot_width/2

    fig.text(0.5, 1-bottom_offset/2, title, ha='center', va='center', fontsize=16)
    ellip_w = 2.0 * p_k / np.max(p_k)

    # Draw marginal distributions
    for d1 in range(D):
        for d2 in range(d1+1, D):
            #subplot(D-1, D-1, d1+1+d2*(D-1))
            cax = fig.add_axes([d1*plot_width+left_offset, (d2-1)*plot_width+bottom_offset, plot_width, plot_width])
            range1 = np.linspace(np.min(X[:,d1]), np.max(X[:,d1]), 18)
            range2 = np.linspace(np.min(X[:,d2]), np.max(X[:,d2]), 18)
            cen_marg, cov_marg, mc_marg = gmm.marg_dist([d1, d2], cen_lst, cov_lst, p_k)
            prob = np.ones((len(range1), len(range2)))
            if draw_pcolor:
                for i in range(len(range1)):
                    X_ = np.concatenate((np.c_[range1], np.ones((len(range2),1))*range2[i]), axis=1)
                    print cen_marg, cov_marg, mc_marg
                    prob[i, :] = gmm.gmm_pdf(X_, cen_marg, cov_marg, mc_marg)
            if draw_ellipses:
                for i in range(len(mc_marg)):
                    x1, x2 = gmm.gauss_ellipse_2d(cen_marg[i], cov_marg[i])
                    plot(x1, x2, color=ellipse_color, linewidth=ellip_w[i], zorder=10)#, alpha=mc_cond[i]/np.max(mc_cond))
            if draw_scatter:
                scatter(X[:,d1], X[:,d2], marker='x', color=scatter_color, linewidths=0.2, zorder=5)
            if draw_pcolor:
                pcolormesh(range1, range2, np.log(prob), vmin=vmin, vmax=vmax)
            grid()
            axis([range1[0], range1[-1], range2[0], range2[-1]])
            if d1==0:
                if labels==None:
                    ylabel(str(d2))
                else:
                    ylabel(labels[d2])
                a,b = yticks()
                yticks([a[1], a[-2]], [str(a[1]), str(a[-2])])
            else:
                a,b = yticks()
                yticks([a[1], a[-2]],['', ''])
            if d2==d1+1:
                if labels==None:
                    xlabel(str(d1))
                else:
                    xlabel(labels[d1])
                a,b = xticks()
                xticks([a[1], a[-2]], [str(a[1]), str(a[-2])])
            else:
                a,b = xticks()
                xticks([a[1], a[-2]],['', ''])
            #cax = fig.add_axes([0.825, 0.1, 0.05, 0.8])
            #cb = colorbar(cax=cax,orientation='vertical')
            #cb.set_label('$log_{10}(p)$')
            #return (cen_lst, cov_lst, p_k, logL, K, max_iter_stored)

