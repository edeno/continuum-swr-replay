# Fit one synthetic data set with mgcv's gam (REML, bs = "ps") for
# compare_regression_to_mgcv.py, and write what that script compares as JSON.
#
# Usage: Rscript compare_regression_to_mgcv.R <data.csv> <kind> <k> <lo> <hi> <grid.csv> <out.json>
#
# kind: poisson_one_smooth (y ~ s(s) + rew, offset log(expo)), poisson_two_smooths
# (y ~ s(s) + s(u) + rew, offset log(expo)), bernoulli_one_smooth (y ~ s(s)),
# gaussian_one_smooth (y ~ s(s)). Every smooth is a cubic P-spline with a
# second-order difference penalty (bs = "ps", m = c(2, 2)) on k basis functions
# whose knots split [lo, hi] into k - 3 equal intervals, as the shared fitter's
# PSplineBasis. The convergence control is tightened: with mgcv's defaults the
# outer search stops early enough to move a log penalty by about 2e-4.

suppressMessages(library(mgcv))
args <- commandArgs(TRUE)
d <- read.csv(args[1])
kind <- args[2]
k <- as.integer(args[3])
lo <- as.numeric(args[4])
hi <- as.numeric(args[5])
grid <- read.csv(args[6])$x
out <- args[7]

h <- (hi - lo) / (k - 3)
kn <- lo + h * seq(-3, k)
control <- gam.control(epsilon = 1e-12, newton = list(conv.tol = 1e-10))

g <- switch(kind,
  poisson_one_smooth = gam(
    y ~ s(s, bs = "ps", k = k, m = c(2, 2)) + rew + offset(log(expo)),
    family = poisson, method = "REML", control = control, data = d,
    knots = list(s = kn)
  ),
  poisson_two_smooths = gam(
    y ~ s(s, bs = "ps", k = k, m = c(2, 2)) + s(u, bs = "ps", k = k, m = c(2, 2)) +
      rew + offset(log(expo)),
    family = poisson, method = "REML", control = control, data = d,
    knots = list(s = kn, u = kn)
  ),
  bernoulli_one_smooth = gam(
    y ~ s(s, bs = "ps", k = k, m = c(2, 2)),
    family = binomial, method = "REML", control = control, data = d,
    knots = list(s = kn)
  ),
  gaussian_one_smooth = gam(
    y ~ s(s, bs = "ps", k = k, m = c(2, 2)),
    family = gaussian, method = "REML", control = control, data = d,
    knots = list(s = kn)
  ),
  stop("unknown kind ", kind)
)

st <- summary(g)
res <- list(
  kind = kind,
  r_version = R.version.string,
  mgcv_version = as.character(packageVersion("mgcv")),
  n_rows = nrow(d),
  y_sum = sum(d$y),
  converged = g$converged,
  outer_iterations = g$outer.info$iter,
  sp = unname(g$sp),
  trace_S = sapply(g$smooth, function(sm) sum(diag(sm$S[[1]]))),
  edf = unname(st$s.table[, "edf"]),
  reml = unname(g$gcv.ubre),
  scale = g$sig2
)
if ("rew" %in% names(coef(g))) {
  i <- which(names(coef(g)) == "rew")
  res$coef_rew <- unname(coef(g)[i])
  res$se_rew <- sqrt(g$Vp[i, i])
}
nd <- data.frame(s = grid, u = grid, rew = 0, expo = 1)
pr <- predict(g, nd, type = "terms", se.fit = TRUE)
res$curve_s <- unname(pr$fit[, "s(s)"])
res$se_curve_s <- unname(pr$se.fit[, "s(s)"])
if (kind == "poisson_two_smooths") {
  res$curve_u <- unname(pr$fit[, "s(u)"])
  res$se_curve_u <- unname(pr$se.fit[, "s(u)"])
}
writeLines(jsonlite::toJSON(res, digits = NA, auto_unbox = TRUE), out)
