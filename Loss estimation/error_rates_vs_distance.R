install.packages("ggplot2")
install.packages("dplyr")
install.packages("readr")

library(readr)
library(reticulate)
library(ggplot2)
library(dplyr)


# Load Python modules
pickle <- import("pickle")
pd <- import("pandas")
builtins <- import_builtins()

# Load the pickle data
dataset_path <- "D:\\GitHub\\MLDS\\machine_learning_mini_projects\\Loss estimation\\data\\dataset.csv"
file_path <- "D:\\GitHub\\MLDS\\machine_learning_mini_projects\\Loss estimation\\all_model_results.pkl"
data <- read_delim(dataset_path, delim = ";", col_names = TRUE)
head(data)
con <- builtins$open(file_path, "rb")
results_p1 <- pickle$load(con)
con$close()

# Extract necessary data
y_true <- results_p1$y_true
y_pred_baseline <- results_p1$y_pred_baseline
y_pred_lr <- results_p1$y_pred_lr
y_pred_rf1 <- results_p1$y_pred_rf1
y_pred_rf2 <- results_p1$y_pred_rf2

distances <- data$Distance

test_indices_python <- results_p1$test_indices  # Get the Python Index object
test_indices_r <- py_to_r(test_indices_python$values)  # Extract values from the pandas Index

# Ensure the indices are in a numeric format
test_indices_r <- as.numeric(test_indices_r)

# Access the 'Distance' column using the indices
distance_test <- data[test_indices_r, 'Distance']
# Print the result
print(distance_test)

# Define bins
min_distance <- min(distances)
max_distance <- max(distances)
bins <- seq(min_distance, max_distance, length.out = 7)  # 6 bins
print("Bins:")
print(bins)

# Create bin labels
bin_labels <- paste0(floor(bins[-length(bins)]), "-", floor(bins[-1]))
print("Bin Labels:")
print(bin_labels)

# Assign distance values to bins
distance_bins <- cut(distances, breaks = bins, labels = bin_labels, include.lowest = TRUE)

# Get percentage of each bin
distance_bin_percentages <- prop.table(table(distance_bins)) * 100
print("Distance Bin Percentages:")
print(distance_bin_percentages)

# Bin the test distances
distance_bins_test <- cut(distance_test$Distance, breaks = bins, labels = bin_labels, include.lowest = TRUE)

# Get test set bin percentages
distance_bin_tsp <- prop.table(table(distance_bins_test)) * 100
print("Distance Bins (Test Set):")
print(distance_bin_tsp)

# Define misclassification rate function
calculate_misclassification_rate <- function(y_true, y_pred, distance_bins) {
  df <- data.frame(y_true = y_true, y_pred = y_pred, distance_bins = distance_bins)
  misclassification_rate <- df %>%
    group_by(distance_bins) %>%
    summarise(misclassification_rate = mean(y_true != y_pred, na.rm = TRUE))

  return(misclassification_rate)
}

# Calculate misclassification rates
misclassification_rate_baseline <- calculate_misclassification_rate(y_true, y_pred_baseline, distance_bins_test)
misclassification_rate_lr <- calculate_misclassification_rate(y_true, y_pred_lr, distance_bins_test)
misclassification_rate_rf1 <- calculate_misclassification_rate(y_true, y_pred_rf1, distance_bins_test)
misclassification_rate_rf2 <- calculate_misclassification_rate(y_true, y_pred_rf2, distance_bins_test)

# Convert distance_bin_tsp to a results_p1 frame for merging
distance_bin_tsp_df <- data.frame(distance_bins = names(distance_bin_tsp), tsp = as.numeric(distance_bin_tsp))

# Merge and compute error rates
compute_error_rates <- function(misclassification_rate_df) {
  merged_df <- merge(misclassification_rate_df, distance_bin_tsp_df, by = "distance_bins", all.x = TRUE)
  merged_df$error_rate <- merged_df$misclassification_rate * merged_df$tsp
  return(merged_df)
}

error_rates_baseline <- compute_error_rates(misclassification_rate_baseline)
error_rates_lr <- compute_error_rates(misclassification_rate_lr)
error_rates_rf1 <- compute_error_rates(misclassification_rate_rf1)
error_rates_rf2 <- compute_error_rates(misclassification_rate_rf2)

# Print error rates
print("Error Rates (Baseline):")
print(error_rates_baseline)
print("Error Rates (Logistic Regression):")
print(error_rates_lr)
print("Error Rates (Random Forest 1):")
print(error_rates_rf1)
print("Error Rates (Random Forest 2):")
print(error_rates_rf2)
# Plot error rates
ggplot() +
  geom_line(data = error_rates_baseline, aes(x = distance_bins, y = error_rate, color = "Baseline", group = 1)) +
  geom_line(data = error_rates_lr, aes(x = distance_bins, y = error_rate, color = "Logistic Regression", group = 1)) +
  geom_line(data = error_rates_rf1, aes(x = distance_bins, y = error_rate, color = "Random Forest 1", group = 1)) +
  geom_line(data = error_rates_rf2, aes(x = distance_bins, y = error_rate, color = "Random Forest 2", group = 1)) +
  labs(x = "Distance (feet)", y = "Error Rate", title = "Error Rate vs. Distance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_manual(values = c("Baseline" = "blue", "Logistic Regression" = "red",
                                "Random Forest 1" = "green", "Random Forest 2" = "purple"))
