* Date of last revision: June 26, 2013

******************************************************	
/*	Innovations for Poverty Action (IPA)
	November 21, 2014
	Replication run using Stata 13.1
	For replication purposes, edited outreg code in
	order to conform to updated syntax. Edits noted
	with **IPA EDIT** below.						*/
******************************************************

clear
clear matrix
clear mata
set mem 300m
set more off
set logtype text

capture log close
log using demogAER.log, replace

*use demogAER1.dta, clear
use "Data/AER merged", clear


**************GEN INTERACTION VARIABLES*************************
gen t_year5 = treatment * year5
gen t_hpa = treatment * hpa
gen t_pwhite=treatment*pwhite
gen t_pblack=treatment*pblack
gen t_page18_39=treatment*page18_39
gen t_ave_hh_sz = treatment*ave_hh_sz
gen t_psch_atlstba = treatment*psch_atlstba
gen t_powner = treatment*powner
gen t_red0=treatment*red0
gen t_poppropurban = treatment*pop_propurban



***********************INCOME**********************

rename median_hhincome medinc
gen medinc_adj = medinc/100000
gen t_medinc_adj = treatment*medinc_adj

*TABLE 6
dprobit gave treatment red0 t_red0 year5 t_year5 hpa t_hpa
//outreg using table6new, replace se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, replace starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***) se bdec(3)

dprobit gave treatment red0 t_red0 pwhite t_pwhite
//outreg using table6new, replace se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, merge starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***)  se bdec(3)

//dprobit gave treatment red0 t_red0 pwhite t_pwhite **IPA EDIT** //IPA note: Inclusion of this code replaced the previous table entries and was otherwise redundant, so was commented out.
//outreg using table6new, replace se bdec(3) /*3aster coefastr*/ 

dprobit gave treatment red0 t_red0 pblack t_pblack
//outreg using table6new, append  se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, merge starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***)  se bdec(3)

dprobit gave treatment red0 t_red0 page18_39 t_page18_39
//outreg using table6new, append  se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, merge starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***)  se bdec(3)

dprobit gave treatment red0 t_red0 ave_hh_sz t_ave_hh_sz
//outreg using table6new, append  se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, merge starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***)  se bdec(3)

dprobit gave treatment red0 t_red0 medinc_adj t_medinc_adj
//outreg using table6new, append  se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, merge starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***)  se bdec(3)

dprobit gave treatment red0 t_red0 powner t_powner
//outreg using table6new, append  se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, merge starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***)  se bdec(3) 

dprobit gave treatment red0 t_red0 psch_atlstba t_psch_atlstba
//outreg using table6new, append  se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, merge starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***)  se bdec(3)

*dprobit gave treatment red0 t_red0 pop_propurban t_poppropurban
*outreg using table6new, append se bdec(3) /*3aster coefastr*/

dprobit gave treatment red0 t_red0 pwhite t_pwhite pblack t_pblack page18_39 t_page18_39 ave_hh_sz t_ave_hh_sz medinc_adj t_medinc_adj powner t_powner psch_atlstba t_psch_atlstba pop_propurban t_poppropurban
//outreg using table6new, append  se bdec(3) /*3aster coefastr*/ **IPA EDIT**
outreg using table6new, merge starloc(1) starlevels(10 5 1) sigsymbols(*, **, ***)  se bdec(3)

log close
