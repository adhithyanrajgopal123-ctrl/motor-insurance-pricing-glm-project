install.packages(c("dplyr", "ggplot2", "tidyr", "MASS", "statmod", 
                   "xgboost", "caret", "pROC", "gridExtra", "scales"))

install.packages("OpenML")
library(OpenML)
library(dplyr)

install.packages("farff")
library(OpenML)
library(dplyr)

freMTPL2freq <- getOMLDataSet(data.id = 41214)$data
freMTPL2sev <- getOMLDataSet(data.id = 41215)$data

str(freMTPL2freq)
dim(freMTPL2freq)

# freMTPL2freq = OpenML dataset ID 41214 (frequency: policy-level data)
freMTPL2freq <- getOMLDataSet(data.id = 41214)$data

# freMTPL2sev = OpenML dataset ID 41215 (severity: individual claim amounts)
freMTPL2sev <- getOMLDataSet(data.id = 41215)$data

# Confirm both loaded
str(freMTPL2freq)
dim(freMTPL2freq)

str(freMTPL2sev)
dim(freMTPL2sev)


# Aggregate severity to policy level
sev_agg <- freMTPL2sev %>%
  group_by(IDpol) %>%
  summarise(ClaimAmount = sum(ClaimAmount), .groups = "drop")

# Merge onto frequency table
df <- freMTPL2freq %>%
  left_join(sev_agg, by = "IDpol") %>%
  mutate(ClaimAmount = ifelse(is.na(ClaimAmount), 0, ClaimAmount))

glimpse(df)

# --- Sanity checks / cleaning ---
summary(df$Exposure)
df <- df %>% filter(Exposure > 0, Exposure <= 1)

table(df$ClaimNb)
df <- df %>% mutate(ClaimNb = pmin(ClaimNb, 4))

df %>% filter(ClaimNb > 0, ClaimAmount == 0) %>% nrow()
df %>% filter(ClaimNb == 0, ClaimAmount > 0) %>% nrow()

quantile(df$ClaimAmount, probs = c(0.95, 0.99, 0.999, 1))
df <- df %>% mutate(ClaimAmount = pmin(ClaimAmount, 200000))

df <- df %>%
  mutate(
    DrivAgeBand = cut(DrivAge, breaks = c(17,22,26,31,41,51,61,71,99), right = FALSE),
    VehAgeBand  = cut(VehAge, breaks = c(-1,1,4,7,10,15,100), right = TRUE)
  )

glimpse(df)

n_distinct(df$IDpol) == nrow(df)

# --- 1. Overall claim frequency & severity summary ---
df %>% summarise(
  n_policies = n(),
  total_exposure = sum(Exposure),
  total_claims = sum(ClaimNb),
  overall_freq = sum(ClaimNb) / sum(Exposure),
  claims_with_amount = sum(ClaimAmount > 0),
  mean_severity = mean(ClaimAmount[ClaimAmount > 0]),
  median_severity = median(ClaimAmount[ClaimAmount > 0])
)

# --- 2. Distribution checks ---
hist(df$ClaimNb, breaks = 0:5, main = "Claim Count Distribution",col="blue")
hist(df$Exposure, breaks = 30, main = "Exposure Distribution",col="red")
hist(log(df$ClaimAmount[df$ClaimAmount > 0]), breaks = 30, 
     main = "Log Claim Amount Distribution (nonzero only)",col="green")

# --- 3. Overdispersion check ---
df %>% summarise(mean_claims = mean(ClaimNb), var_claims = var(ClaimNb))

# --- 4. Frequency by rating factor ---
freq_by_factor <- function(data, factor_var) {
  data %>%
    group_by(across(all_of(factor_var))) %>%
    summarise(freq = sum(ClaimNb)/sum(Exposure), exposure = sum(Exposure), .groups = "drop")
}

freq_by_factor(df, "DrivAgeBand")
freq_by_factor(df, "VehAgeBand")
freq_by_factor(df, "Area")
freq_by_factor(df, "VehGas")
freq_by_factor(df, "VehBrand")


# Check the DrivAgeBand NA edge case
df %>% filter(is.na(DrivAgeBand)) %>% select(DrivAge) %>% summary()

# Check Area vs Density correlation (flagged above)
df %>% group_by(Area) %>% summarise(mean_density = mean(Density))

# Correlation matrix among numeric factors
numeric_vars <- df %>% select(VehPower, VehAge, DrivAge, BonusMalus, Density)
cor(numeric_vars, use = "complete.obs")

# BonusMalus vs frequency
df %>%
  mutate(BM_band = cut(BonusMalus, breaks = seq(50, 230, by = 20))) %>%
  group_by(BM_band) %>%
  summarise(freq = sum(ClaimNb)/sum(Exposure), n = n())


df %>%
  mutate(BM_band = cut(BonusMalus, breaks = seq(50, 230, by = 20), include.lowest = TRUE)) %>%
  group_by(BM_band) %>%
  summarise(freq = sum(ClaimNb)/sum(Exposure), n = n())

df %>% filter(is.na(DrivAgeBand)) %>% select(DrivAge) %>% summary()

# Correlation matrix among numeric factors
numeric_vars <- df %>% select(VehPower, VehAge, DrivAge, BonusMalus, Density)
cor(numeric_vars, use = "complete.obs")

# Area vs Density check
df %>% group_by(Area) %>% summarise(mean_density = mean(Density))

df <- df %>% mutate(log_Density = log(Density))

# quick check of the relationship shape
df %>%
  mutate(DensityBand = cut(log_Density, breaks = 6)) %>%
  group_by(DensityBand) %>%
  summarise(freq = sum(ClaimNb)/sum(Exposure), n = n())

df <- df %>% mutate(log_Density = log(Density))

"log_Density" %in% names(df)

df %>%
  mutate(DensityBand = cut(log_Density, breaks = 6)) %>%
  group_by(DensityBand) %>%
  summarise(freq = sum(ClaimNb)/sum(Exposure), n = n())

df %>%
  mutate(DensityBand = cut(log_Density, breaks = 6)) %>%
  group_by(DensityBand) %>%
  summarise(freq = sum(ClaimNb)/sum(Exposure), n = n())

# --- Train/Test Split ---
set.seed(42)
train_idx <- sample(1:nrow(df), size = 0.7 * nrow(df))
train <- df[train_idx, ]
test  <- df[-train_idx, ]

dim(train)
dim(test)

# --- Frequency Model: Poisson GLM ---
glm_freq <- glm(
  ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + log_Density + 
    VehBrand + VehGas + Region,
  offset = log(Exposure),
  family = poisson(link = "log"),
  data = train
)

summary(glm_freq)

# --- Dispersion check ---
deviance(glm_freq) / df.residual(glm_freq)

install.packages("car")
library(car)
vif(glm_freq)


# --- Severity Model: Gamma GLM ---
train_sev <- train %>% filter(ClaimAmount > 0)

glm_sev <- glm(
  ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + log_Density + 
    VehBrand + VehGas + Region,
  family = Gamma(link = "log"),
  data = train_sev
)

summary(glm_sev)

install.packages("statmod")
library(statmod)

glm_tweedie <- glm(
  ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + log_Density + 
    VehBrand + VehGas + Region,
  offset = log(Exposure),
  family = tweedie(var.power = 1.5, link.power = 0),
  data = train
)

summary(glm_tweedie)

install.packages("xgboost")
library(xgboost)

# Prepare model matrix (XGBoost needs numeric matrix, not factors directly)
train_matrix <- model.matrix(~ VehPower + VehAge + DrivAge + BonusMalus + log_Density + 
                               VehBrand + VehGas + Region - 1, data = train)
test_matrix  <- model.matrix(~ VehPower + VehAge + DrivAge + BonusMalus + log_Density + 
                               VehBrand + VehGas + Region - 1, data = test)

dtrain <- xgb.DMatrix(data = train_matrix, label = train$ClaimNb, 
                      base_margin = log(train$Exposure))
dtest  <- xgb.DMatrix(data = test_matrix, label = test$ClaimNb, 
                      base_margin = log(test$Exposure))

# --- Replace the old xgboost() call with this xgb.train() version ---
params <- list(
  objective = "count:poisson",
  max_depth = 4,
  eta = 0.1
)

xgb_freq <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain),
  verbose = 1
)

# --- Predictions on test set ---
test$pred_glm <- predict(glm_freq, newdata = test, type = "response")
test$pred_xgb <- predict(xgb_freq, newdata = dtest)

# --- Quick sanity check: compare total predicted vs actual claims ---
test %>% summarise(
  actual_claims = sum(ClaimNb),
  glm_predicted = sum(pred_glm),
  xgb_predicted = sum(pred_xgb)
)

# Function to compute Gini coefficient for insurance pricing models
# Based on ordering by predicted risk, then measuring how well actual claims concentrate
gini_insurance <- function(actual, predicted, exposure) {
  df_gini <- data.frame(actual = actual, predicted = predicted, exposure = exposure) %>%
    arrange(predicted) %>%
    mutate(
      cum_exposure = cumsum(exposure) / sum(exposure),
      cum_actual = cumsum(actual) / sum(actual)
    )
  
  # Gini = 2 * area between Lorenz curve and diagonal
  n <- nrow(df_gini)
  gini <- 1 - 2 * sum(df_gini$cum_actual) / n + 1/n
  return(gini)
}

gini_glm <- gini_insurance(test$ClaimNb, test$pred_glm, test$Exposure)
gini_xgb <- gini_insurance(test$ClaimNb, test$pred_xgb, test$Exposure)

cat("GLM Gini:", gini_glm, "\n")
cat("XGBoost Gini:", gini_xgb, "\n")

lift_table <- function(actual, predicted, exposure, n_bins = 10) {
  data.frame(actual = actual, predicted = predicted, exposure = exposure) %>%
    mutate(decile = ntile(predicted, n_bins)) %>%
    group_by(decile) %>%
    summarise(
      avg_predicted = sum(predicted) / sum(exposure),
      avg_actual = sum(actual) / sum(exposure),
      n = n()
    )
}

lift_glm <- lift_table(test$ClaimNb, test$pred_glm, test$Exposure)
lift_xgb <- lift_table(test$ClaimNb, test$pred_xgb, test$Exposure)

print(lift_glm)
print(lift_xgb)


lift_table_weighted <- function(actual, predicted, exposure, n_bins = 10) {
  data.frame(actual = actual, predicted = predicted, exposure = exposure) %>%
    mutate(rate = predicted / exposure) %>%
    arrange(rate) %>%
    mutate(
      cum_exposure = cumsum(exposure),
      decile = ceiling(cum_exposure / sum(exposure) * n_bins),
      decile = pmin(decile, n_bins)  # guard against rounding pushing into bin 11
    ) %>%
    group_by(decile) %>%
    summarise(
      avg_predicted = sum(predicted) / sum(exposure),
      avg_actual = sum(actual) / sum(exposure),
      total_exposure = sum(exposure),
      .groups = "drop"
    )
}

lift_glm <- lift_table_weighted(test$ClaimNb, test$pred_glm, test$Exposure)
lift_xgb <- lift_table_weighted(test$ClaimNb, test$pred_xgb, test$Exposure)

print(lift_glm)
print(lift_xgb)


library(ggplot2)
library(tidyr)

# Reshape for plotting
plot_lift <- function(lift_data, title) {
  lift_data %>%
    pivot_longer(cols = c(avg_predicted, avg_actual), names_to = "type", values_to = "rate") %>%
    ggplot(aes(x = decile, y = rate, color = type)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    scale_x_continuous(breaks = 1:10) +
    labs(title = title, x = "Decile (low to high predicted risk)", y = "Claim Frequency") +
    theme_minimal()
}

plot_lift(lift_glm, "GLM Lift Chart")
plot_lift(lift_xgb, "XGBoost Lift Chart")


ae_by_factor <- function(data, factor_var, pred_col) {
  data %>%
    group_by(across(all_of(factor_var))) %>%
    summarise(
      Actual = sum(ClaimNb),
      Expected = sum(.data[[pred_col]]),
      AE_ratio = Actual / Expected,
      Exposure = sum(Exposure),
      .groups = "drop"
    )
}

# Overall A/E for both models
test %>% summarise(
  Actual = sum(ClaimNb),
  Expected_GLM = sum(pred_glm),
  AE_GLM = Actual / Expected_GLM,
  Expected_XGB = sum(pred_xgb),
  AE_XGB = Actual / Expected_XGB
)

# A/E by DrivAgeBand
ae_by_factor(test, "DrivAgeBand", "pred_glm")
ae_by_factor(test, "DrivAgeBand", "pred_xgb")

# A/E by VehAgeBand
ae_by_factor(test, "VehAgeBand", "pred_glm")
ae_by_factor(test, "VehAgeBand", "pred_xgb")

# A/E by Region
ae_by_factor(test, "Region", "pred_glm")
ae_by_factor(test, "Region", "pred_xgb")

# GLM: is actual significantly different from expected for [17,22)?
young_glm <- test %>% filter(DrivAgeBand == "[17,22)")
poisson.test(x = sum(young_glm$ClaimNb), T = sum(young_glm$pred_glm))

# XGBoost: same test
poisson.test(x = sum(young_glm$ClaimNb), T = sum(young_glm$pred_xgb))


# --- Relativities table from GLM coefficients ---
# For a log-link Poisson GLM, exp(coefficient) gives the multiplicative relativity

coefs <- summary(glm_freq)$coefficients
relativities <- data.frame(
  Variable = rownames(coefs),
  Coefficient = coefs[, "Estimate"],
  Relativity = exp(coefs[, "Estimate"]),
  P_value = coefs[, "Pr(>|z|)"]
)

# Filter to significant terms for a clean summary table
relativities %>% filter(P_value < 0.05) %>% arrange(desc(Relativity))

# Pick a few representative policy profiles and show old vs new predicted premium
sample_policies <- test %>% 
  sample_n(5) %>%
  select(IDpol, DrivAge, VehAge, BonusMalus, VehBrand, Region, ClaimNb, pred_glm, pred_xgb)

print(sample_policies)


# Illustrate the compounding effect of BonusMalus explicitly
bm_effect <- exp(0.02253169 * (130 - 50))  # comparing BM=130 to baseline BM=50
cat("Relativity for BonusMalus=130 vs 50:", round(bm_effect, 2), "\n")

sample_policies <- sample_policies %>%
  mutate(pct_diff = (pred_xgb - pred_glm) / pred_glm * 100)

print(sample_policies)

# BonusMalus: 130 vs baseline 50
bm_effect <- exp(0.02253169 * (130 - 50))
cat("BonusMalus=130 vs 50:", round(bm_effect, 2), "x baseline frequency\n")

# DrivAge: 55 vs 25 (illustrating the counter-intuitive positive coefficient)
age_effect <- exp(0.00643752 * (55 - 25))
cat("DrivAge=55 vs 25:", round(age_effect, 2), "x baseline frequency\n")

# VehAge: 1 year vs 10 years (illustrating newer=riskier finding)
vehage_effect <- exp(-0.03848740 * (1 - 10))
cat("VehAge=1 vs 10:", round(vehage_effect, 2), "x baseline frequency\n")

# Combined worst-case profile: young driver + high BonusMalus + new car
combined_effect <- exp(0.00643752*(55-25) + 0.02253169*(130-50) + -0.03848740*(1-10))
cat("Combined high-risk profile:", round(combined_effect, 2), "x baseline frequency\n")





install.packages("writexl")
library(writexl)




# Option A: Full cleaned dataset (~676k rows, may be a large file, slower to open)
write_xlsx(df, "Documents/GitHub/motor-insurance-pricing-glm/data/cleaned_policy_data.xlsx")

dir.create("GitHub/motor-insurance-pricing-glm/data", recursive = TRUE, showWarnings = FALSE)
dir.create("GitHub/motor-insurance-pricing-glm/outputs", recursive = TRUE, showWarnings = FALSE)

# Option B: A representative sample (recommended for a lighter, more usable file)
set.seed(1)
df_sample <- df[sample(1:nrow(df), 10000), ]
write_xlsx(df_sample, "Documents/GitHub/motor-insurance-pricing-glm/data/cleaned_policy_data_sample.xlsx")

getwd()

library(writexl)

# Sample dataset (recommended)
set.seed(1)
df_sample <- df[sample(1:nrow(df), 10000), ]
write_xlsx(df_sample, "GitHub/motor-insurance-pricing-glm/data/cleaned_policy_data_sample.xlsx")

# Model outputs
model_outputs <- list(
  "Predictions" = test %>% select(IDpol, ClaimNb, Exposure, pred_glm, pred_xgb),
  "Relativities" = relativities %>% filter(P_value < 0.05) %>% arrange(desc(Relativity)),
  "AE_by_DrivAge" = ae_by_factor(test, "DrivAgeBand", "pred_glm"),
  "AE_by_VehAge" = ae_by_factor(test, "VehAgeBand", "pred_glm"),
  "Lift_GLM" = lift_glm,
  "Lift_XGBoost" = lift_xgb
)

write_xlsx(model_outputs, "GitHub/motor-insurance-pricing-glm/outputs/model_results.xlsx")

file.exists("GitHub/motor-insurance-pricing-glm/data/cleaned_policy_data_sample.xlsx")
file.exists("GitHub/motor-insurance-pricing-glm/outputs/model_results.xlsx")
