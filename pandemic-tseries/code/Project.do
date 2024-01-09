import excel "D:\Masters\spring 23\Applied Econometrics and Time Series Analysis\PROJECT\APPLIED_ECON_PROJ.xlsx", sheet("Sheet1") cellrange(A1:N154) firstrow
//Create a log function of population
generate logpop = log(pop)
//Making the heatplot
correlate
return list
matrix corrmat = r(C)
heatplot corrmat, values(format(%4.3f) size(tiny)) color(hcl diverging, intensity(.7)) xlabel(,labsize(small) angle(90)) title("HEATPLOT")
//Histograms of all the variables
histogram obes_perc , freq kdensity
histogram pop , freq kdensity
histogram avg_inc , freq kdensity
histogram alccon , freq kdensity
histogram tob_avg , freq kdensity
histogram pop_pov , freq kdensity
histogram logpop , freq kdensity
//Regression
regress obes_perc year pop avg_inc alccon tob_avg pop_pov w mw sw s
regress obes_perc year logpop avg_inc alccon tob_avg pop_pov w mw sw s

//Create var for pandemic
generate cov = year > 2019
//Create yearly time trend
generate time_tr = year - 2018

regress obes_perc time_tr logpop log_inc alccon tob_avg pop_pov cov w mw sw s
// Multicollinearity
vif
// Homoskedasticity
estat imtest
regress cov time_tr logpop log_inc alccon tob_avg pop_pov w mw sw s
predict res, residuals
correlate res time_tr, covariance
correlate cov time_tr, covariance
ivregress 2sls obes_perc logpop avg_inc alccon tob_avg pop_pov w mw sw s (cov = time_tr)