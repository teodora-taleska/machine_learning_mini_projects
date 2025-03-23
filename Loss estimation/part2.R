install.packages("ggplot2")
install.packages("dplyr")
install.packages("readr")

library(readr)
library(reticulate)
library(ggplot2)
library(dplyr)
library(stats)

py_require('pandas')

pickle <- import("pickle")
pd <- import("pandas")
builtins <- import_builtins()


dataset_path <- "D:\\GitHub\\MLDS\\machine_learning_mini_projects\\Loss estimation\\data\\dataset.csv"
file_path <- "D:\\GitHub\\MLDS\\machine_learning_mini_projects\\Loss estimation\\all_model_results.pkl"
data <- read_delim(dataset_path, delim = ";", col_names = TRUE)
con <- builtins$open(file_path, "rb")
results_p1 <- pickle$load(con)
con$close()


# PART 1 - Does error depend on Distance?

distances <- data$Distance
test_indices_python <- results_p1$test_indices  # Get the Python Index object
test_indices_r <- py_to_r(test_indices_python$values)  # Extract values from the pandas Index
test_indices_r <- as.numeric(test_indices_r)

distance_test <- data[test_indices_r, 'Distance']
distance_test <- distance_test$Distance

calculate_spearman <- function(errors, distance) {
  result <- cor.test(errors, distance, method = "spearman", exact = FALSE)
  return(list(spearman_corr = result$estimate, p_value = result$p.value))
}

spearman_baseline <- calculate_spearman(results_p1$errors_baseline, distance_test)
spearman_lr <- calculate_spearman(results_p1$errors_lr, distance_test)
spearman_rf1 <- calculate_spearman(results_p1$errors_rf1, distance_test)
spearman_rf2 <- calculate_spearman(results_p1$errors_rf2, distance_test)

print("Baseline:")
print(paste("Spearman correlation:", spearman_baseline$spearman_corr, "p-value:", spearman_baseline$p_value))

print("Logistic Regression:")
print(paste("Spearman correlation:", spearman_lr$spearman_corr, "p-value:", spearman_lr$p_value))

print("Random Forest 1:")
print(paste("Spearman correlation:", spearman_rf1$spearman_corr, "p-value:", spearman_rf1$p_value))

print("Random Forest 2:")
print(paste("Spearman correlation:", spearman_rf2$spearman_corr, "p-value:", spearman_rf2$p_value))


# PART 2 - Estimating model performance on true distribution

log_loss <- function(y_true, y_pred) {
  # Clip predicted probabilities to avoid log(0) or log(1)
  y_pred <- pmax(pmin(y_pred, 1 - 1e-10), 1e-10)
  
  # Convert y_true to one-hot encoding
  n_classes <- ncol(y_pred) 
  y_true_one_hot <- model.matrix(~ as.factor(y_true) - 1, data = data.frame(y_true = y_true))
  
  # Ensure y_true_one_hot has the same number of columns as y_pred
  if (ncol(y_true_one_hot) < n_classes) {
    # Pad y_true_one_hot with zeros for missing classes
    y_true_one_hot <- cbind(y_true_one_hot, matrix(0, nrow = nrow(y_true_one_hot), ncol = n_classes - ncol(y_true_one_hot)))
  } else if (ncol(y_true_one_hot) > n_classes) {
    # Pad y_pred with zeros for missing classes
    y_pred <- cbind(y_pred, matrix(0, nrow = nrow(y_pred), ncol = ncol(y_true_one_hot) - n_classes))
  }
  
  loss <- -mean(rowSums(y_true_one_hot * log(y_pred)))
  
  return(loss)
}

accuracy_score <- function(y_true, y_pred) {
  correct <- sum(y_true == y_pred)
  accuracy <- correct / length(y_true)
  return(accuracy)
}

competition_types <- data$Competition
competition_types_test <- competition_types[test_indices_r]

# Methodology - Re-weighting errors
calculate_weighted_metrics <- function(y_true, y_pred, y_pred_proba, competition_types, weights) {
  log_loss_total <- 0
  accuracy_total <- 0

  for (comp_type in names(weights)) {
    comp_mask <- (competition_types == comp_type)
    if (sum(comp_mask) > 0) { # check if there are any samples for this competition type
      y_true_comp <- y_true[comp_mask]
      y_pred_comp <- y_pred[comp_mask]
      y_pred_proba_comp <- y_pred_proba[comp_mask, ]

      log_loss_comp <- log_loss(y_true_comp, y_pred_proba_comp)
      accuracy_comp <- accuracy_score(y_true_comp, y_pred_comp)
      
      log_loss_total <- log_loss_total + log_loss_comp * weights[[comp_type]]
      accuracy_total <- accuracy_total + accuracy_comp * weights[[comp_type]]
    }
  }

  return(list(log_loss = log_loss_total, accuracy = accuracy_total))
}

weights <- list(
  NBA = 0.6,
  EURO = 0.1,
  SLO1 = 0.1,
  U14 = 0.1,
  U16 = 0.1
)

weighted_metrics <- list(
  baseline = calculate_weighted_metrics(results_p1$y_true, results_p1$y_pred_baseline, results_p1$y_pred_proba_baseline, competition_types_test, weights),
  lr = calculate_weighted_metrics(results_p1$y_true, results_p1$y_pred_lr, results_p1$y_pred_proba_lr, competition_types_test, weights),
  rf1 = calculate_weighted_metrics(results_p1$y_true, results_p1$y_pred_rf1, results_p1$y_pred_proba_rf1, competition_types_test, weights),
  rf2 = calculate_weighted_metrics(results_p1$y_true, results_p1$y_pred_rf2, results_p1$y_pred_proba_rf2, competition_types_test, weights)
)


print("Weighted Metrics:")
for (model in names(weighted_metrics)) {
  metrics <- weighted_metrics[[model]]
  cat(model, ": Log-Loss =", round(metrics$log_loss, 4), ", Accuracy =", round(metrics$accuracy, 4), "\n")
}


models <- c("Baseline", "LR", "RF (Flat CV)", "RF (Nested CV)")
accuracy_values <- sapply(weighted_metrics, function(x) x$accuracy)
log_loss_values <- sapply(weighted_metrics, function(x) x$log_loss)

# Cap log loss values at 2.5 for visualization
log_loss_values <- pmin(log_loss_values, 2.5)

plot_data <- data.frame(
  Model = rep(models, 2),
  Metric = c(rep("Accuracy", length(models)), rep("Log Loss", length(models))),
  Value = c(accuracy_values, log_loss_values)
)

# Create the plot
ggplot(plot_data, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  geom_text(
    aes(label = round(Value, 2)),
    position = position_dodge(width = 0.7),
    vjust = -0.5,
    size = 3
  ) +
  scale_fill_manual(values = c("Accuracy" = "lightblue", "Log Loss" = "lightpink")) +
  labs(
    title = "Model Comparison: Accuracy and Log Loss",
    x = "Models",
    y = "Score"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 10),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 8),
    legend.text = element_text(size = 8)
  ) +
  ylim(0, 1.3)













