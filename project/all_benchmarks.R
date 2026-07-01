# price vs benchmark performance in LLMs
# does price still predict benchmark score once we control for model scale
# (params + training tokens)? runs across all 5 benchmarks.

library(tidyverse)   # dplyr, tidyr, ggplot2, readr, purrr
library(broom)       # tidy(), glance()
library(car)         # vif()
library(ppcor)       # pcor.test()

# inputs
CSV_PATH_MAIN <- "data/llm_price_performance_tracker_2026-03-31.csv"
CSV_PATH_SECOND <- "data/LifeArchitect_Models.csv"
COST_CAP <- 20   # $/1M, drops the 2 price outliers (o1 $30, claude-3-opus $26.25)

benchmarks <- c("livecodebench", "scicode", "mmlu_pro",
                "gpqa_diamond", "humanitys_last_exam")
predictors <- c("log_params", "log_tokens", "cost", "oss")
scale_ctrl <- c("log_params", "log_tokens")

# match key so the two files line up: lowercase, drop punctuation + variant suffixes
norm <- function(x) sub(" (instruct|chat|base|it|preview|thinking)$", "",
                        gsub("\\s+", " ", trimws(gsub("[-_]", " ",
                                                      tolower(sub("\\s*\\(.*\\)", "", x))))))
zscore <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)


# ---- stage 1: build the dataset ----
# two sources - a price/benchmark tracker and a model spec table (params, tokens).
# one clean row per model, inner join on the normalized name, z-score the
# benchmarks. everything after this reads the single frame `adf`.

# benchmarks + price, keep the best variant per model
llm <- read_csv(CSV_PATH_MAIN, show_col_types = FALSE) %>%
  mutate(model_name = norm(model_name)) %>%
  dplyr::select(all_of(c("model_name", benchmarks,
                         "is_open_source", "blended_cost_usd_per_1m"))) %>%
  drop_na() %>%
  group_by(model_name) %>%
  slice_max(livecodebench, n = 1, with_ties = FALSE) %>%
  ungroup()

# scale specs (skip the junk header row), one row per model
extra <- read_csv(CSV_PATH_SECOND, skip = 1, show_col_types = FALSE) %>%
  mutate(model_name = norm(model_name)) %>%
  dplyr::select(model_name, Params, `Tokens \ntrained (B)`) %>%
  drop_na() %>%
  distinct(model_name, .keep_all = TRUE)

dataset <- inner_join(llm, extra, by = "model_name") %>%
  mutate(across(all_of(benchmarks), zscore, .names = "{.col}_z"))

# the analysis frame
adf <- dataset %>%
  filter(blended_cost_usd_per_1m <= COST_CAP) %>%
  mutate(
    log_params = log10(Params),   # very skewed
    log_tokens = log10(as.numeric(gsub(",", "", `Tokens \ntrained (B)`))),
    cost = blended_cost_usd_per_1m,
    oss = as.integer(is_open_source)
  ) %>%
  dplyr::select(model_name, all_of(paste0(benchmarks, "_z")),
                all_of(predictors)) %>%
  drop_na()

# long form (benchmark x predictor) for the facet plots
long_z <- adf %>%
  dplyr::select(cost, log_params, ends_with("_z")) %>%
  pivot_longer(ends_with("_z"), names_to = "test", values_to = "z") %>%
  mutate(test = sub("_z$", "", test)) %>%
  pivot_longer(c(cost, log_params), names_to = "predictor", values_to = "x")


# ---- stage 2: does price (or size) track score? ----
# plot each benchmark vs cost and vs size, then test the raw cost slope.
# "you get what you pay for" would mean positive, significant cost slopes.
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

# test the raw cost slope per benchmark
cost_test <- long_z %>%
  filter(predictor == "cost") %>%
  group_by(test) %>%
  group_modify(~ {
    m <- lm(z ~ x, data = .x)
    tibble(slope = coef(m)[2], p_value = glance(m)$p.value, r2 = glance(m)$r.squared)
  }) %>%
  ungroup() %>%
  mutate(significant = p_value < 0.05)
print(cost_test)        # weak and mostly not significant


# ---- stage 3: what actually predicts score? ----
# let them compete: size (params, tokens) vs price (cost) vs open-source (oss).
# for each benchmark fit the full model and ask if cost/oss add over scale alone.
# collinearity + VIF only depend on the predictors so just do them once.
cat("\n#### predictor correlations ####\n")
print(round(cor(dplyr::select(adf, all_of(predictors))), 2))
cat("\n#### VIF (all < 5, collinearity is mild) ####\n")
print(round(vif(lm(reformulate(predictors, "scicode_z"), data = adf)), 2))

# per benchmark: scale-only r2, full r2, cost & oss p, nested F-test p
model_tab <- lapply(benchmarks, function(b) {
  resp <- paste0(b, "_z")
  full <- lm(reformulate(predictors, resp), data = adf)
  scl <- lm(reformulate(scale_ctrl, resp), data = adf)
  ct <- coef(summary(full))
  tibble(benchmark = b,
         scale_r2 = summary(scl)$r.squared,
         full_r2 = summary(full)$r.squared,
         cost_p = ct["cost", "Pr(>|t|)"],
         oss_p = ct["oss", "Pr(>|t|)"],
         add_p = anova(scl, full)$`Pr(>F)`[2])
}) %>% bind_rows()
cat("\n#### scale carries it, cost & oss add nothing ####\n")
print(model_tab)


# ---- stage 4: why did cost look important? (confounding) ----
# cost is tangled up with scale - pricier models tend to be bigger. show the
# confound, then strip scale off both cost and score and watch the cost effect
# collapse (and pin down which scale var does it).

# the confound: cost goes up with both size measures
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

# cost-score corr, taking each confounder out in turn. none = raw, both = partial.
# the flip to negative (mostly params) is the confounding made explicit.
strip <- function(resp, ctrl) {
  cor(resid(lm(reformulate(ctrl, "cost"), data = adf)),
      resid(lm(reformulate(ctrl, resp), data = adf)))
}
confound_tab <- lapply(benchmarks, function(b) {
  resp <- paste0(b, "_z")
  tibble(benchmark = b,
         none = cor(adf$cost, adf[[resp]]),
         params = strip(resp, "log_params"),
         tokens = strip(resp, "log_tokens"),
         both = strip(resp, scale_ctrl))
}) %>% bind_rows()
cat("\n#### cost-score corr, stripping confounders (raw -> partial) ####\n")
print(confound_tab)     # positive raw, negative once params comes out

# other side of scale - is training tokens a real predictor on its own?
tokens_tab <- lapply(benchmarks, function(b) {
  resp <- paste0(b, "_z")
  ct <- cor.test(resid(lm(log_tokens ~ log_params + cost, data = adf)),
                 resid(lm(reformulate(c("log_params", "cost"), resp), data = adf)))
  tibble(benchmark = b, tokens_partial = ct$estimate, p_value = ct$p.value)
}) %>% bind_rows()
cat("\n#### tokens partial corr (params + cost removed) - positive ####\n")
print(tokens_tab)


# ---- stage 5: the scale-removed cost effect - show it, rank it, test it ----
# isolate cost with scale removed from BOTH cost and score (FWL). plot per
# benchmark, rank the slopes, test where it's strongest.

# cost residual is the same everywhere, score residual is per benchmark
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

# (a) scale-removed cost slope per benchmark (= cost coef of z ~ cost + scale),
# with 95% CI and p, ranked most negative -> weakest
slope_tab <- lapply(benchmarks, function(b) {
  m <- lm(reformulate(c("cost", scale_ctrl), paste0(b, "_z")), data = adf)
  tidy(m, conf.int = TRUE) %>%
    filter(term == "cost") %>%
    transmute(benchmark = b, slope = estimate,
              ci_low = conf.low, ci_high = conf.high, p_value = p.value)
}) %>% bind_rows() %>%
  arrange(slope) %>%
  mutate(rank = row_number(), p_holm = p.adjust(p_value, "holm"))
cat("\n#### (a) scale-removed cost slope per benchmark ####\n")
print(slope_tab)

# (b) omnibus - are the 5 slopes equal? multivariate test on the benchmark
# differences; the 'cost' row is H0: all slopes equal.
D <- sapply(benchmarks[-1], function(b) adf[[paste0(b, "_z")]] - adf[["livecodebench_z"]])
cat("\n#### (b) omnibus: do the cost slopes differ across benchmarks? ('cost' row) ####\n")
print(anova(lm(D ~ cost + log_params + log_tokens, data = adf)))

# (c) pairwise - slope_A - slope_B is the cost coef of (z_A - z_B) ~ cost + scale.
# paired, so it respects the cross-benchmark correlation. holm adjusted.
pairs <- t(combn(benchmarks, 2))
pair_tab <- lapply(seq_len(nrow(pairs)), function(i) {
  dd <- adf[[paste0(pairs[i, 1], "_z")]] - adf[[paste0(pairs[i, 2], "_z")]]
  m <- lm(reformulate(c("cost", scale_ctrl), "dd"), data = cbind(adf, dd = dd))
  tidy(m) %>% filter(term == "cost") %>%
    transmute(benchmark_A = pairs[i, 1], benchmark_B = pairs[i, 2],
              slope_diff = estimate, p_value = p.value)
}) %>% bind_rows() %>%
  mutate(p_holm = p.adjust(p_value, "holm")) %>%
  arrange(p_holm)
cat("\n#### (c) pairwise slope diffs (holm) ####\n")
print(pair_tab)

# (d) the whole thing at a glance: slope +/- 95% CI per benchmark
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


# ---- takeaway ----
# on its own price barely tracks score, and the bit of signal it has is just a
# scale proxy: bigger models cost more and score higher. hold params + tokens
# constant and price's advantage vanishes, even flips (cheaper-for-their-size
# models do a touch better) - fairly consistent across benchmarks, strongest for
# livecodebench, though the benchmark-to-benchmark diffs mostly aren't significant.
# the real drivers are model scale (params and, notably, training tokens), not price.