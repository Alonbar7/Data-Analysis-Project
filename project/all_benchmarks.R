# =============================================================================
# Do you get what you pay for? Price vs. benchmark performance in LLMs
# -----------------------------------------------------------------------------
# Question: once we account for model scale (parameters, training tokens), does a
# model's PRICE still predict how well it scores on capability benchmarks?
# Road map: build data -> surface look (cost & size) -> model what drives score
# -> show cost only LOOKED important (confounding) -> rank the residual effect
# -> conclude. Everything runs across all five benchmarks.
# =============================================================================

# ---- libraries ----
library(tidyverse)   # dplyr, tidyr, ggplot2, readr, purrr
library(broom)       # tidy(), glance()
library(car)         # vif()
library(ppcor)       # pcor.test()

# ---- inputs ----
CSV_PATH_MAIN   <- "data/llm_price_performance_tracker_2026-03-31.csv"
CSV_PATH_SECOND <- "data/LifeArchitect_Models.csv"
COST_CAP        <- 20   # $/1M; drops 2 price outliers (o1 $30, claude-3-opus $26.25)

benchmarks <- c("livecodebench", "scicode", "mmlu_pro",
                "gpqa_diamond", "humanitys_last_exam")
predictors <- c("log_params", "log_tokens", "cost", "oss")
scale_ctrl <- c("log_params", "log_tokens")

# match key for the two sources: lower-case, strip punctuation & variant suffixes
norm <- function(x) sub(" (instruct|chat|base|it|preview|thinking)$", "",
                        gsub("\\s+", " ", trimws(gsub("[-_]", " ",
                                                      tolower(sub("\\s*\\(.*\\)", "", x))))))
zscore <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

# =============================================================================
# STAGE 1 - BUILD THE DATASET
# Two sources: a price/benchmark tracker and a model-spec table (params, tokens).
# Keep one clean row per model (strongest variant), inner-join on the normalized
# name, z-score the benchmarks, then build ONE analysis frame: drop price
# outliers, add predictors on sensible scales, keep complete cases. Every later
# stage reads this single frame (`adf`).
# =============================================================================

# source A: benchmarks + price; keep the strongest variant per model
llm <- read_csv(CSV_PATH_MAIN, show_col_types = FALSE) %>%
  mutate(model_name = norm(model_name)) %>%
  dplyr::select(all_of(c("model_name", benchmarks,
                         "is_open_source", "blended_cost_usd_per_1m"))) %>%
  drop_na() %>%
  group_by(model_name) %>%
  slice_max(livecodebench, n = 1, with_ties = FALSE) %>%
  ungroup()

# source B: scale specs (skip junk header row); one row per model
extra <- read_csv(CSV_PATH_SECOND, skip = 1, show_col_types = FALSE) %>%
  mutate(model_name = norm(model_name)) %>%
  dplyr::select(model_name, Params, `Tokens \ntrained (B)`) %>%
  drop_na() %>%
  distinct(model_name, .keep_all = TRUE)

# inner-join, z-score the benchmarks
dataset <- inner_join(llm, extra, by = "model_name") %>%
  mutate(across(all_of(benchmarks), zscore, .names = "{.col}_z"))

# the single analysis frame
adf <- dataset %>%
  filter(blended_cost_usd_per_1m <= COST_CAP) %>%
  mutate(
    log_params = log10(Params),                                    # heavily skewed
    log_tokens = log10(as.numeric(gsub(",", "", `Tokens \ntrained (B)`))),
    cost       = blended_cost_usd_per_1m,
    oss        = as.integer(is_open_source)
  ) %>%
  dplyr::select(model_name, all_of(paste0(benchmarks, "_z")),
                all_of(predictors)) %>%
  drop_na()

# long form (benchmark x predictor) for the faceted plots
long_z <- adf %>%
  dplyr::select(cost, log_params, ends_with("_z")) %>%
  pivot_longer(ends_with("_z"), names_to = "test", values_to = "z") %>%
  mutate(test = sub("_z$", "", test)) %>%
  pivot_longer(c(cost, log_params), names_to = "predictor", values_to = "x")

# =============================================================================
# STAGE 2 - SURFACE LOOK: DOES PRICE (OR SIZE) TRACK SCORE?
# Plot each benchmark (z) against cost and against size, then formally test the
# raw cost slope. "You get what you pay for" predicts positive, significant cost
# slopes; we expect size to look far stronger than price.
# =============================================================================
slopes <- long_z %>%
  group_by(test, predictor) %>%
  summarise(slope = coef(lm(z ~ x))[2], .groups = "drop") %>%
  mutate(label = sprintf("slope = %+.4f", slope))

print(
  ggplot(long_z, aes(x, z)) +
    geom_point(alpha = 0.5, size = 1.4, colour = "#2c7fb8") +
    geom_smooth(method = "lm", se = FALSE, colour = "#d95f0e", linewidth = 0.8) +
    geom_text(data = slopes, aes(Inf, Inf, label = label),
              hjust = 1.1, vjust = 1.5, size = 3, inherit.aes = FALSE) +
    facet_grid(test ~ predictor, scales = "free_x",
               labeller = labeller(predictor = c(cost = "Blended cost (USD/1M)",
                                                 log_params = "log10(Params, B)"))) +
    labs(title = "Benchmark (z) sensitivity to cost vs model size",
         subtitle = "Rows = benchmark; slope per panel (x-scales differ; cost outliers removed)",
         x = NULL, y = "Score (z)") +
    theme_minimal(base_size = 11)
)

# formal test of the RAW cost slope per benchmark
cost_test <- long_z %>%
  filter(predictor == "cost") %>%
  group_by(test) %>%
  group_modify(~ {
    m <- lm(z ~ x, data = .x)
    tibble(slope = coef(m)[2], p_value = glance(m)$p.value, r2 = glance(m)$r.squared)
  }) %>%
  ungroup() %>%
  mutate(significant = p_value < 0.05)
print(cost_test)        # cost slopes are weak and mostly non-significant

# =============================================================================
# STAGE 3 - WHAT ACTUALLY PREDICTS SCORE?
# Let the predictors compete: size (log params, log tokens), price (cost) and
# open-source (oss). For each benchmark, fit the full model and ask whether
# price/oss add anything over scale alone. (Collinearity & VIF depend only on the
# predictors, so they are computed once.)
# =============================================================================
cat("\n#### predictor collinearity (shared across benchmarks) ####\n")
print(round(cor(dplyr::select(adf, all_of(predictors))), 2))
cat("\n#### VIF (all < 5 -> collinearity mild; depends only on predictors) ####\n")
print(round(vif(lm(reformulate(predictors, "scicode_z"), data = adf)), 2))

# per-benchmark: scale-only R2, full R2, cost & oss significance, and the nested
# F-test p-value (does cost + oss add over scale?). One compact table.
model_tab <- lapply(benchmarks, function(b) {
  resp <- paste0(b, "_z")
  full <- lm(reformulate(predictors, resp), data = adf)
  scl  <- lm(reformulate(scale_ctrl, resp), data = adf)
  ct   <- coef(summary(full))
  tibble(benchmark = b,
         scale_r2 = summary(scl)$r.squared,
         full_r2  = summary(full)$r.squared,
         cost_p   = ct["cost", "Pr(>|t|)"],
         oss_p    = ct["oss",  "Pr(>|t|)"],
         add_p    = anova(scl, full)$`Pr(>F)`[2])
}) %>% bind_rows()
cat("\n#### predictors of score: scale carries it; cost & oss add nothing ####\n")
print(model_tab)        # scale_r2 ~ full_r2; cost_p, oss_p, add_p all non-significant

# =============================================================================
# STAGE 4 - WHY DID COST LOOK IMPORTANT? (CONFOUNDING)
# Cost is entangled with scale: pricier models tend to be bigger. We show the
# confound, then peel scale off cost and score and watch the apparent cost effect
# collapse - and pin down which scale variable is responsible.
# =============================================================================

# the confound: cost rises with both size measures
print(
  adf %>%
    pivot_longer(c(log_params, log_tokens), names_to = "predictor", values_to = "value") %>%
    ggplot(aes(value, cost)) +
    geom_point(alpha = 0.6, colour = "#2c7fb8") +
    geom_smooth(method = "lm", se = TRUE, colour = "#d95f0e") +
    facet_wrap(~ predictor, scales = "free_x") +
    labs(title = "Cost is confounded with scale - pricier models are bigger",
         y = "Cost (USD / 1M)") +
    theme_minimal(base_size = 12)
)

# decomposition: cost-score correlation, removing each confounder in turn.
# "none" is the raw correlation; "both" is the partial. The flip to negative -
# driven mainly by params - is the confounding made explicit.
strip <- function(resp, ctrl) {
  cor(resid(lm(reformulate(ctrl, "cost"), data = adf)),
      resid(lm(reformulate(ctrl, resp),  data = adf)))
}
confound_tab <- lapply(benchmarks, function(b) {
  resp <- paste0(b, "_z")
  tibble(benchmark = b,
         none   = cor(adf$cost, adf[[resp]]),
         params = strip(resp, "log_params"),
         tokens = strip(resp, "log_tokens"),
         both   = strip(resp, scale_ctrl))
}) %>% bind_rows()
cat("\n#### cost-score correlation, stripping confounders (raw -> partial) ####\n")
print(confound_tab)     # positive raw -> negative once params (mainly) is removed

# the other side of scale: is training tokens a genuine predictor on its own?
tokens_tab <- lapply(benchmarks, function(b) {
  resp <- paste0(b, "_z")
  ct <- cor.test(resid(lm(log_tokens ~ log_params + cost, data = adf)),
                 resid(lm(reformulate(c("log_params", "cost"), resp), data = adf)))
  tibble(benchmark = b, tokens_partial = ct$estimate, p_value = ct$p.value)
}) %>% bind_rows()
cat("\n#### tokens partial correlation (params + cost removed): genuinely positive ####\n")
print(tokens_tab)

# =============================================================================
# STAGE 5 - THE SCALE-REMOVED COST EFFECT: SHOW IT, RANK IT, TEST IT
# Having established the confound, isolate cost's effect with scale removed from
# BOTH cost and score (Frisch-Waugh-Lovell). Visualize it for every benchmark,
# rank the slopes, and test where it is strongest.
# =============================================================================

# residuals: cost is the same across benchmarks; score residual is per benchmark
cost_resid <- resid(lm(cost ~ log_params + log_tokens, data = adf))
resid_df <- lapply(benchmarks, function(b) {
  sr <- resid(lm(reformulate(scale_ctrl, paste0(b, "_z")), data = adf))
  tibble(benchmark = b, cost_resid = cost_resid, score_resid = sr)
}) %>% bind_rows()

print(
  ggplot(resid_df, aes(cost_resid, score_resid)) +
    geom_point(alpha = 0.6, size = 1.4, colour = "#2c7fb8") +
    geom_smooth(method = "lm", se = TRUE, colour = "#d95f0e") +
    facet_wrap(~ benchmark, ncol = 3, scales = "free_y") +
    labs(title = "After removing scale, cost's effect on score is negative",
         x = "Cost (residual, scale removed)",
         y = "Benchmark z (residual, scale removed)") +
    theme_minimal(base_size = 11)
)

# (a) per-benchmark scale-removed cost slope (= cost coef of z ~ cost + scale),
#     with 95% CI and p, ranked strongest (most negative) -> weakest
slope_tab <- lapply(benchmarks, function(b) {
  m <- lm(reformulate(c("cost", scale_ctrl), paste0(b, "_z")), data = adf)
  tidy(m, conf.int = TRUE) %>%
    filter(term == "cost") %>%
    transmute(benchmark = b, slope = estimate,
              ci_low = conf.low, ci_high = conf.high, p_value = p.value)
}) %>% bind_rows() %>%
  arrange(slope) %>%
  mutate(rank = row_number(), p_holm = p.adjust(p_value, "holm"))
cat("\n#### (a) scale-removed cost slope per benchmark (strongest -> weakest) ####\n")
print(slope_tab)

# (b) omnibus: are the five slopes equal? Multivariate test on the
#     benchmark-difference outcomes; the 'cost' row tests H0: all slopes equal.
D     <- sapply(benchmarks[-1], function(b) adf[[paste0(b, "_z")]] - adf[["livecodebench_z"]])
cat("\n#### (b) omnibus: do the cost slopes differ across benchmarks? ('cost' row) ####\n")
print(anova(lm(D ~ cost + log_params + log_tokens, data = adf)))

# (c) pairwise: slope_A - slope_B is the 'cost' coef of (z_A - z_B) ~ cost + scale,
#     a paired test that respects the cross-benchmark correlation. Holm-adjusted.
pairs <- t(combn(benchmarks, 2))
pair_tab <- lapply(seq_len(nrow(pairs)), function(i) {
  dd <- adf[[paste0(pairs[i, 1], "_z")]] - adf[[paste0(pairs[i, 2], "_z")]]
  m  <- lm(reformulate(c("cost", scale_ctrl), "dd"), data = cbind(adf, dd = dd))
  tidy(m) %>% filter(term == "cost") %>%
    transmute(benchmark_A = pairs[i, 1], benchmark_B = pairs[i, 2],
              slope_diff = estimate, p_value = p.value)
}) %>% bind_rows() %>%
  mutate(p_holm = p.adjust(p_value, "holm")) %>%
  arrange(p_holm)
cat("\n#### (c) pairwise slope differences (Holm-adjusted) ####\n")
print(pair_tab)

# (d) the comparison at a glance: slope +/- 95% CI per benchmark
print(
  ggplot(slope_tab, aes(slope, reorder(benchmark, slope))) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.18, colour = "#2c7fb8") +
    geom_point(colour = "#2c7fb8", size = 2.6) +
    labs(title = "Scale-removed cost effect by benchmark (slope +/- 95% CI)",
         subtitle = "More negative = stronger cost penalty; CI crossing 0 = not individually significant",
         x = "Cost slope (z per USD/1M, scale removed)", y = NULL) +
    theme_minimal(base_size = 12)
)

# =============================================================================
# CONCLUSION
# On its own, price barely tracks benchmark score, and the little signal it has
# is a SCALE PROXY: bigger models cost more and score higher. Once parameters and
# training tokens are held constant, price's apparent advantage vanishes and even
# reverses (cheaper-for-their-size models do slightly better) - consistently
# across benchmarks, strongest for livecodebench, though the benchmark-to-
# benchmark differences are mostly not significant. The real, independent drivers
# of score are model scale - parameters and, notably, training tokens - not how
# much the model costs.
# =============================================================================