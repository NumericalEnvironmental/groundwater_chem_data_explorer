#######################################################################
#
# gama_data_exploration.py
#
# by Walt McNab
#
# process and analyze groundwater quality data sets from the California
# Groundwater Ambient Monitoring and Assessment program
#
#######################################################################

from numpy import *
from pandas import *
from datetime import *
from scipy.stats.mstats import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import *

# supporting functions

def ExtractData(file_name, analytes):

    # read tab-delimited text file as dataframe; process and extract analytes subset pivot table
    df = read_csv(file_name, sep='\t')
    df['CHEMICAL'] = ['Sodium' if isinstance(x,float) else x for x in df['CHEMICAL']]   # NA abbreviation for sodium is problematic and is subtituted away here
    mask = df['CHEMICAL'].isin(analytes)                                                # limit chemicals in dataframe to those on analytes list
    df = df[mask]
    df['DATE'] = to_datetime(df['DATE'])                                                # process sample dates
    ppm_match = array(df['UNITS']=='MG/L')                                              # process units (convert ug/L to mg/L)
    df['CONC'] = ppm_match*df['RESULT'] + (1-ppm_match)*0.001*df['RESULT'] 
    nd_match = array(df['QUALIFIER']=='<')                                              # process non-detects
    df['CONC'] = nd_match*0.5*df['CONC'] + (1-nd_match)*df['CONC']
    df = df[(df['CONC']>0.)]

    # convert to pivot table (columns = analytes) and return
    df_pivot = pivot_table(df, values='CONC', index=['WELL NAME','DATE','APPROXIMATE LONGITUDE', 'APPROXIMATE LATITUDE', 'DATASET'], columns=['CHEMICAL'])
    df_pivot.reset_index(inplace=True)
    return df_pivot

def pca_loadings(num_pca_comps, pca_set, analytes_set, file_name):
    # process PCA set and write component loadings to file
    loadings_set = PCA(n_components=num_pca_comps).fit(pca_set)
    loadings = loadings_set.components_
    loadings_df = DataFrame(analytes_set)
    loadings_df.columns = ['analytes']
    loadings_vals_df = DataFrame(loadings.transpose())
    col_label = []
    for i in xrange(num_pca_comps): col_label.append('comp_' + str(i))
    loadings_vals_df.columns = col_label
    loadings_df = concat([loadings_df, loadings_vals_df], axis=1)
    loadings_df.to_csv(file_name)
    print ' '
    print 'Principal components contribution to variance, file = ', file_name
    print loadings_set.explained_variance_ratio_                                # indicate how much variance each PC explains (FYI feature)


# main routine

def Gama():

    # definitions: these are hard-wired in in this script, but could easily be read in from text files

    counties = ['Tehama', 'Glenn', 'Butte', 'Colusa', 'Sutter', 'Yuba', 'Yolo', 'Placer', 'Solano', 'Sacramento', 'SanJoaquin',
        'Stanislaus', 'Merced', 'Madera', 'Fresno', 'Kings', 'Tulare', 'Kern']
    analytes = ['AS', 'B', 'BA', 'CU', 'CR', 'MN', 'SO4', 'CL', 'Sodium', 'MG', 'CA', 'K', 'ALKB', 'NO3', 'ZN']
    analytes_subset = ['SO4', 'CL', 'Sodium', 'MG', 'CA', 'K', 'ALKB', 'NO3']   # for trend analysis principal component analysis
    start_date = '01/01/1970'
    start_t = to_datetime(start_date)                           # start of temporal analysis period
    t0 = start_t.toordinal()
    min_reports = 10                                            # minimum number of detections for trend analysis
    num_pca_comps = 4                                           # number of components to be used for principal component analysis

    for k, county in enumerate(counties):       # process county groundwater chemistry data

        # GAMA data sets are available for download by county; all use the same naming convention
        print 'Working on : ', county
        file_name = 'gama_all_' + county + '.txt'    
        county_df = ExtractData(file_name, analytes)

        # extract and record median values by analyte and by individual well as separate data frames, then concatenate into single data frame 
        county_medians_df = county_df.groupby(['WELL NAME', 'DATASET', 'APPROXIMATE LONGITUDE', 'APPROXIMATE LATITUDE']).median()       # assumption is that well names, plus associate latitude and longitude, are sufficient to delineate unique wells
        county_medians_df.reset_index(inplace=True)
        wells_list = county_medians_df[['WELL NAME', 'APPROXIMATE LONGITUDE', 'APPROXIMATE LATITUDE', 'DATASET']].values    # extract unique well designations
        if not k: all_medians_df = county_medians_df
        else: all_medians_df = concat([all_medians_df, county_medians_df], axis=0)

        # compute temporal trends, per well, by Theil-Sen slope

        slope = empty((len(wells_list), len(analytes)), float)          # create empty placeholder matrix
        slope[:, :] = NaN    

        for i, well_ID in enumerate(wells_list):            # wells_list = list of all wells in all counties (delineated by lat-long, as needed)

            # select data sets from wells after the earliest cutoff date (t0)
            well_history_df = county_df[(county_df['WELL NAME']==well_ID[0])
                & (county_df['APPROXIMATE LONGITUDE']==well_ID[1])
                & (county_df['APPROXIMATE LATITUDE']==well_ID[2])
                & (county_df['DATE'].apply(lambda x: x.toordinal())>=t0)]

            if len(well_history_df):                    # for those wells with sampling history after date t0
                t = array(well_history_df['DATE'].apply(lambda x: x.toordinal()) - t0)           # elapsed time (days) since t0, for trend analysis
                for j, chem in enumerate(analytes):
                    num_detect = len(well_history_df[chem]) - isnan(well_history_df[chem]).sum()
                    if num_detect >= min_reports:           # conduct sen slope analysis if a minimum number of samples exist after t0
                        y = array(well_history_df[chem])
                        slope[i, j] = theilslopes(y, t)[0]
                        

        # create data frame containing the sen slope results, per well, per analyte
        county_sen_df = DataFrame(wells_list)
        county_sen_df.columns = ['WELL NAME', 'APPROXIMATE LONGITUDE', 'APPROXIMATE LATITUDE', 'DATASET']
        slope_matrix_df = DataFrame(slope)
        slope_matrix_df.columns = analytes
        county_sen_df = concat([county_sen_df, slope_matrix_df], axis=1)
        county_sen_df.dropna(axis=0, how='all', inplace=True)                         # county version
        if not k: all_sen_df = county_sen_df
        else: all_sen_df = concat([all_sen_df, county_sen_df], axis=0)

    # add wiggles to location of non-EDF wells (well locations are centered on township-range-section grid, so need to separate them for viewing in GIS)
    loc_flag = array(all_medians_df['DATASET']!='EDF')
    all_medians_df['APPROXIMATE LONGITUDE'] += loc_flag * random.uniform(-0.5, 0.5, size=len(all_medians_df)) * 0.02
    all_medians_df['APPROXIMATE LATITUDE'] += loc_flag * random.uniform(-0.5, 0.5, size=len(all_medians_df)) * 0.02
    loc_flag = array(all_sen_df['DATASET']!='EDF')
    all_sen_df['APPROXIMATE LONGITUDE'] += loc_flag * random.uniform(-0.5, 0.5, size=len(all_sen_df)) * 0.02
    all_sen_df['APPROXIMATE LATITUDE'] += loc_flag * random.uniform(-0.5, 0.5, size=len(all_sen_df)) * 0.02

    # write output for medians and sen trends sets
    all_medians_df.to_csv('all_medians_df.csv', index=False)
    all_sen_df.to_csv('all_sen_df.csv', index=False) 

    # conduct principal components analysis (PCA) on medians set
    compressed_set_df = all_medians_df.copy()
    compressed_set_df.dropna(axis=0, how='any', inplace=True)
    pca_set = log10(compressed_set_df[analytes].values)         # pca set = log concentrations of analytes in a fully filled matrix (well x analyte)
    pca = PCA(n_components=num_pca_comps).fit_transform(pca_set)
    for i in xrange(num_pca_comps):
        col_label = 'comp_' + str(i)
        compressed_set_df[col_label] = pca[:,i]
    pca_loadings(num_pca_comps, pca_set, analytes, 'medians_pca_loadings.csv')          # compute PCA loadings and write to file

    # conduct K-means cluster analysis of medians set
    k_means = KMeans(init='k-means++', n_clusters=10, n_init=25)                    # K-means cluster analysis
    z = k_means.fit_predict(pca_set)                                                # use same PCA set defined above
    compressed_set_df['kmeans_group'] = z
    compressed_set_df.to_csv('pca.csv')
    centroids_df = DataFrame(k_means.cluster_centers_, columns=analytes)            # note cluster centroids and write to output file
    centroids_df.to_csv('centroids.csv')

    # conduct PCA on sen trends set
    report = linspace(start=5., stop=95., num=19, endpoint=True)
    percent = empty((len(analytes), len(report)))
    num_trends = zeros(len(analytes), int)
    for i, chem in enumerate(analytes):
        trends = array(all_sen_df[chem].dropna())
        num_trends[i] = len(trends)
        percent[i] = percentile(trends, report)

    # process trend histograms and write to output file
    trend_matrix_df = DataFrame(percent)
    trend_matrix_df.columns = DataFrame((report.astype(int)).astype(str))
    trend_histogram_df = DataFrame(analytes)
    trend_histogram_df.columns = ['analytes']
    trend_histogram_df['num_trends'] = num_trends
    trend_histogram_df = concat([trend_histogram_df, trend_matrix_df], axis=1)
    trend_histogram_df.to_csv('trend_histograms.csv')

    # conduct rank-based PCA analysis (distribution of slopes about zero is strongly non-normal, so a non-parametric approach is needed)
    reduced_sen_df = all_sen_df[['WELL NAME', 'APPROXIMATE LONGITUDE', 'APPROXIMATE LATITUDE', 'DATASET'] + analytes_subset]
    reduced_sen_df.dropna(axis=0, how='any', inplace=True)
    for chem in analytes_subset: reduced_sen_df[chem] = rankdata(array(reduced_sen_df[chem]))
    rank_pca_set = reduced_sen_df[analytes_subset].values
    rank_pca = PCA(n_components=num_pca_comps).fit_transform(rank_pca_set)
    for i in xrange(num_pca_comps):
        col_label = 'comp_' + str(i)
        reduced_sen_df[col_label] = rank_pca[:,i]
    reduced_sen_df.to_csv('rank_trend.csv')
    pca_loadings(num_pca_comps, rank_pca_set, analytes_subset, 'rank_trend_pca_loadings.csv')          # compute PCA loadings and write to file

    print 'Done.'


### run script ###

Gama()
