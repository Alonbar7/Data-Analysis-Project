# =============================================================================
# Do you get what you pay for? Price vs. benchmark performance in LLMs
# -----------------------------------------------------------------------------
# Question: once we account for model scale (parameters, training tokens), does a
# model's PRICE still predict how well it scores on capability benchmarks?
# Road map: build data -> look at cost, then size -> model what drives score ->
# explain why cost only LOOKED important -> conclude.
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

# match key for the two sources: lower-case, strip punctuation & variant suffixes
norm <- function(x) sub(" (instruct|chat|base|it|preview|thinking)$", "",
                        gsub("\\s+", " ", trimws(gsub("[-_]", " ",
                                                      tolower(sub("\\s*\\(.*\\)", "", x))))))
zscore <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

# =============================================================================
# STAGE 1 - BUILD THE DATASET
# Two sources: a price/benchmark tracker and a model-spec table (params, tokens).
# Keep one clean row per model (strongest variant), inner-join on the normalized
# name, and z-score each benchmark so the five sit on a common scale.
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

# inner-join -> models present in both sources, then z-score the benchmarks
dataset <- inner_join(llm, extra, by = "model_name") %>%
  mutate(across(all_of(benchmarks), zscore, .names = "{.col}_z"))

# analysis frame: drop price outliers and add predictors on sensible scales
adf <- dataset %>%
  filter(blended_cost_usd_per_1m <= COST_CAP) %>%
  mutate(
    log_params = log10(Params),                                    # heavily skewed
    log_tokens = log10(as.numeric(gsub(",", "", `Tokens \ntrained (B)`))),
    cost       = blended_cost_usd_per_1m,
    oss        = as.integer(is_open_source)
  )

# long form for the faceted plots: one row per (model, benchmark)
long_z <- adf %>%
  dplyr::select(cost, log_params, ends_with("_z")) %>%
  pivot_longer(ends_with("_z"), names_to = "test", values_to = "z") %>%
  mutate(test = sub("_z$", "", test))

# =============================================================================
# STAGE 2 - DOES PRICE TRACK PERFORMANCE?
# First look: each benchmark (z) vs. cost, with a per-benchmark slope and a
# formal test. "You get what you pay for" predicts positive, significant slopes.
# =============================================================================
cost_slopes <- long_z %>%
  group_by(test) %>%
  summarise(slope = coef(lm(z ~ cost))[2], .groups = "drop") %>%
  mutate(label = sprintf("slope = %+.4f", slope))

print(
  ggplot(long_z, aes(cost, z)) +
    geom_point(alpha = 0.5, size = 1.6, colour = "#2c7fb8") +
    geom_smooth(method = "lm", se = FALSE, colour = "#d95f0e", linewidth = 0.8) +
    geom_text(data = cost_slopes, aes(Inf, Inf, label = label),
              hjust = 1.1, vjust = 1.5, size = 3.4, inherit.aes = FALSE) +
    facet_wrap(~ test, ncol = 3) +
    labs(title = "Benchmark (z) vs blended cost",
         x = "Cost (USD / 1M)", y = "Score (z)") +
    theme_minimal(base_size = 12)
)

# is the cost slope actually significant for each benchmark?
cost_test <- long_z %>%
  group_by(test) %>%
  group_modify(~ {
    m <- lm(z ~ cost, data = .x)
    tibble(slope = coef(m)[2], p_value = glance(m)$p.value, r2 = glance(m)$r.squared)
  }) %>%
  ungroup() %>%
  mutate(significant = p_value < 0.05)
print(cost_test)        # slopes are weak and mostly non-significant

# =============================================================================
# STAGE 3 - DOES SIZE TRACK PERFORMANCE?
# Same view against model size (log10 params). If scale is the real driver these
# slopes should be clearly positive - unlike the cost slopes above.
# =============================================================================
size_slopes <- long_z %>%
  group_by(test) %>%
  summarise(slope = coef(lm(z ~ log_params))[2], .groups = "drop") %>%
  mutate(label = sprintf("slope = %+.4f", slope))

print(
  ggplot(long_z, aes(log_params, z)) +
    geom_point(alpha = 0.5, size = 1.6, colour = "#2c7fb8") +
    geom_smooth(method = "lm", se = FALSE, colour = "#d95f0e", linewidth = 0.8) +
    geom_text(data = size_slopes, aes(Inf, Inf, label = label),
              hjust = 1.1, vjust = 1.5, size = 3.4, inherit.aes = FALSE) +
    facet_wrap(~ test, ncol = 3) +
    labs(title = "Benchmark (z) vs model size",
         x = expression(log[10]~"(Params, B)"), y = "Score (z)") +
    theme_minimal(base_size = 12)
)

# =============================================================================
# STAGE 4 - WHAT ACTUALLY PREDICTS SCORE?
# Focus on one benchmark (SciCode) and let candidate predictors compete:
# size (log params, log tokens), price (cost) and open-source (oss). Goal: find
# which carry real, INDEPENDENT explanatory power once they control each other.
# =============================================================================
mdf   <- adf %>% dplyr::select(scicode_z, log_params, log_tokens, cost, oss) %>% drop_na()
preds <- c("log_params", "log_tokens", "cost", "oss")

# collinearity sanity check among the predictors
print(round(cor(mdf[preds]), 2))

# marginal power: each predictor on its own
single <- lapply(preds, function(p) {
  m <- lm(reformulate(p, "scicode_z"), data = mdf)
  glance(m) %>% transmute(predictor = p, r2 = r.squared, p_value = p.value)
}) %>% bind_rows()
print(single)           # scale vars dominate; cost & oss explain little

# full model: do coefficients survive controlling for each other?
full <- lm(scicode_z ~ log_params + log_tokens + cost + oss, data = mdf)
print(summary(full))
print(vif(full))        # all < 5 -> collinearity mild, coefficients interpretable

# does price/oss add anything beyond scale alone?
scale_only <- lm(scicode_z ~ log_params + log_tokens, data = mdf)
print(anova(scale_only, full))   # non-significant F -> they add nothing

# let AIC pick the model, then eyeball diagnostics
best <- step(full, direction = "both", trace = 0)
print(summary(best))             # keeps scale, drops cost & oss
par(mfrow = c(1, 1)); plot(best); par(mfrow = c(1, 1))

# =============================================================================
# STAGE 5 - WHY DID COST LOOK IMPORTANT?
# Cost is entangled with scale: pricier models tend to be bigger. Show the raw
# positive cost-score link flip sign once scale is held constant, quantify it
# with a partial correlation, and pin down which scale variable is responsible.
# =============================================================================

# cost correlates with size -> it is partly just a scale proxy
print(round(cor(mdf[c("cost", "log_params", "log_tokens", "oss")]), 2))

# the sign flip: cost's coefficient alone vs. with scale controlled
sign_flip <- bind_rows(
  tidy(lm(scicode_z ~ cost, data = mdf))                           %>% mutate(model = "cost alone"),
  tidy(lm(scicode_z ~ cost + log_params + log_tokens, data = mdf)) %>% mutate(model = "cost + scale")
) %>% filter(term == "cost") %>% dplyr::select(model, estimate, p.value)
print(sign_flip)        # positive alone, negative once scale is in

# formal partial correlation: cost vs score, holding scale constant
raw_r   <- cor(mdf$cost, mdf$scicode_z)
partial <- pcor.test(mdf$cost, mdf$scicode_z, mdf[c("log_params", "log_tokens")])
cat(sprintf("cost-score correlation:  raw %+.3f  ->  partial %+.3f (p = %.3f)\n",
            raw_r, partial$estimate, partial$p.value))

# visualize the confound: cost rises with both size measures
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

# payoff plot: with scale removed from BOTH, cost's residual link is negative
mdf <- mdf %>% mutate(
  cost_resid  = resid(lm(cost      ~ log_params + log_tokens, data = mdf)),
  score_resid = resid(lm(scicode_z ~ log_params + log_tokens, data = mdf))
)
print(
  ggplot(mdf, aes(cost_resid, score_resid)) +
    geom_point(alpha = 0.6, colour = "#2c7fb8") +
    geom_smooth(method = "lm", se = TRUE, colour = "#d95f0e") +
    labs(title = "After removing scale, cost's effect on SciCode is negative",
         x = "Cost (residual, scale removed)",
         y = "SciCode z (residual, scale removed)") +
    theme_minimal(base_size = 12)
)

# which confounder does the work? strip one at a time
strip <- function(ctrl) {
  cr <- resid(lm(reformulate(ctrl, "cost"),      data = mdf))
  sr <- resid(lm(reformulate(ctrl, "scicode_z"), data = mdf))
  cor(cr, sr)
}
cat(sprintf("cost-score corr | nothing %+.3f | params %+.3f | tokens %+.3f | both %+.3f\n",
            raw_r, strip("log_params"), strip("log_tokens"),
            strip(c("log_params", "log_tokens"))))   # params does most of it

# sanity check the other side: is training tokens a genuine predictor? (yes)
tok_partial <- with(mdf, cor.test(resid(lm(log_tokens ~ log_params + cost)),
                                  resid(lm(scicode_z  ~ log_params + cost))))
cat(sprintf("tokens partial correlation (params+cost removed): %+.3f (p = %.4f)\n",
            tok_partial$estimate, tok_partial$p.value))

# =============================================================================
# CONCLUSION
# On its own, price barely tracks benchmark score, and the little signal it has
# is a SCALE PROXY: bigger models cost more and score higher. Hold size and
# training tokens constant and price's apparent advantage vanishes - even
# reverses (cheaper-for-their-size models do slightly better). The real,
# independent drivers of score are model scale - parameters and, notably,
# training tokens - not how much the model costs.
# =============================================================================