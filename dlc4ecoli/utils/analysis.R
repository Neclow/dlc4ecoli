library(brms)
library(envalysis)
library(ggdist)
library(ggplot2)

folders <- c("dlc", "dlc", "dlc", "of", "of")
regressands <- c("travel", "delta_area", "near_food", "of_mean", "of_var")

labels <- c(
  "Distance travelled\n(px/px; normalised)",
  "Change in body area\n(px^2/px; normalised)",
  "Time spent near\nfood source (%)",
  "Optical flow mean",
  "Optical flow variance"
)

load_data <- function(regressand, folder) {
  d <- na.omit(
    read.csv(
      paste(
        "data/", folder, "/summary_", regressand, ".csv",
        sep = ""
      )
    )
  )

  d$group <- as.factor(
    ifelse(
      d$camera == 2,
      "Control",
      ifelse(
        d$camera == 1,
        "E. coli",
        NA
      )
    )
  )

  return(d)
}

compare_models <- function(
    d, families, regressand,
    adapt_delta = 0.99, max_treedepth = 15, seed = 42) {
  models <- list()

  for (j in seq_along(families)) {
    name <- paste(families[[j]]$family, families[[j]]$link, sep = "_")

    brm_model <- brm(
      as.formula(
        paste(
          regressand,
          "~ group + (1 | Day)"
        )
      ),
      data = d,
      control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth),
      family = families[[j]],
      seed = seed,
      silent = 2
    )

    models[[name]] <- brm_model
  }

  return(models)
}

save_results <- function(best_model, comp, regressand) {
  # Save model
  saveRDS(
    best_model,
    paste("data/brms/model_", regressand, ".Rds", sep = "")
  )

  # Save summary and predictions
  write.csv(
    predict(best_model),
    paste("data/brms/preds_brms_", regressand, ".csv", sep = "")
  )

  write.csv(
    summary(best_model)$fixed,
    paste("data/brms/results_brms_", regressand, ".csv", sep = "")
  )

  write.csv(
    comp,
    paste("data/brms/loo_compare_brms_", regressand, ".csv", sep = "")
  )
}


plot_results <- function(model, regressand) {
  # BRMS convergence plot
  pdf(paste("img/brms_", regressand, ".pdf", sep = ""))

  plot(model)

  dev.off()

  # BRMS posterior predictive check
  pdf(
    paste("img/brms_ppc_", regressand, ".pdf", sep = ""),
    width = 2.63,
    height = 2.63
  )

  pp_check(model) + theme_publish()

  dev.off()
}


compute_ate <- function(d, model, regressand, label) {
  data_control <- data.frame(group = "Control", Day = unique(d$Day))
  data_treatment <- data.frame(group = "E. coli", Day = unique(d$Day))

  # Generate predictions for both groups

  pred_control <- posterior_epred(model, newdata = data_control)
  pred_treatment <- posterior_epred(model, newdata = data_treatment)

  ate <- mean(pred_treatment) / mean(pred_control)

  # Print the ATE
  cat(paste("Estimated ATE: ", ate), "\n")

  # Compute ratios for each posterior sample
  ate_samples <- rowMeans(pred_treatment) / rowMeans(pred_control)

  # Compute credible intervals
  ate_ci <- quantile(ate_samples, probs = c(0.025, 0.975))

  # Print the results
  cat(paste("Mean ATE: ", mean(ate_samples)), "\n")
  cat(paste("95% Credible Interval: [", ate_ci[1], ", ", ate_ci[2], "]"), "\n")


  # Plot it
  ate_data <- data.frame(ATE = rowMeans(pred_treatment) / rowMeans(pred_control))

  ggplot(ate_data, aes(x = ATE)) +
    ggdist::stat_halfeye(colour = "black", fill = "#029e73") +
    theme_publish(base_size = 10) +
    labs(
      x = label,
      y = "Density"
    )

  ggsave(
    paste("img/brms_ate_", regressand, ".pdf", sep = ""),
    width = 2.2,
    height = 2.2
  )
}


for (i in seq_along(regressands)) {
  # Load data
  cat("Loading data...", "\n")
  d <- load_data(regressands[i], folders[i])

  if (
    paste("loo_compare_brms_", regressands[i], ".csv", sep = "") %in%
      list.files("data/brms")
  ) {
    cat("Skipping", regressands[i], "\n")
    next
  }

  if (regressands[i] == "near_food") {
    families <- list(
      # Beta doesn't work for x = 0
      # so use zero-inflated version
      gaussian(),
      zero_inflated_beta(link = "logit")
    )
  } else {
    families <- list(
      # Assume beta because x in (0, 1)
      # for dist travelled and delta_area
      gaussian(link = "identity"),
      Beta(link = "logit")
    )
  }

  cat("Running models...", "\n")
  models <- compare_models(d, families, regressands[i])

  comp <- loo_compare(x = Map(loo, models))

  best_model_name <- rownames(comp)[1]

  best_model <- models[[best_model_name]]

  cat("Save summary and predictions...", "\n")
  save_results(best_model, comp, regressands[i])

  cat("Estimating ATEs...", "\n")
  compute_ate(d, best_model, regressands[i], labels[i])

  cat("Making plots...", "\n")
  plot_results(best_model, regressands[i])

  cat("Done", "\n")
}
