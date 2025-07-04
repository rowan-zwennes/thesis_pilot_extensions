# --- 1. SETUP ---
# install.packages("readxl")
library(readxl)

#' Process MSE Data to Compute Relative Errors and Rankings
#'
#' This function takes a data frame of model names and their corresponding MSE values 
#' across multiple datasets and computes relative MSEs, rankings, and summary statistics 
#' (mean and standard deviation). It returns a formatted summary table useful for model comparison.
#'
#' @param input_df A data frame where the first column contains model names and 
#' the remaining columns contain MSE values for different datasets.
#'
#' @return A data frame containing:
#' \describe{
#'   \item{Model}{Model names (from the first column of the input).}
#'   \item{<dataset columns>}{Relative MSEs for each dataset.}
#'   \item{mean}{Mean of relative MSEs per model.}
#'   \item{std_dev}{Standard deviation of relative MSEs per model.}
#'   \item{avg_rank}{Mean rank based on original MSEs per dataset.}
#'   \item{std_rank}{Standard deviation of ranks.}
#' }
#' 
#' #' @export
process_mse_data <- function(input_df) {
  
  # --- a. PREPARE DATA ---
  # Separate model names from the numeric MSE data
  model_names <- input_df[[1]]
  original_mse <- input_df[, -1] # Keep original MSEs for ranking
  
  # --- b. CALCULATE RELATIVE MSE ---
  # For each column (dataset), divide all values by the minimum value in that column
  relative_mse <- as.data.frame(lapply(original_mse, function(col) {
    col / min(col, na.rm = TRUE)
  }))
  
  # --- c. CALCULATE RANKS ---
  # For each column, rank models based on original MSE.
  # ties.method = "min" handles ties as requested (e.g., 1, 2, 2, 4)
  ranks <- as.data.frame(lapply(original_mse, function(col) {
    rank(col, ties.method = "min", na.last = "keep")
  }))
  
  # --- d. CALCULATE THE 4 NEW SUMMARY COLUMNS ---
  # Calculate mean and standard deviation of the RELATIVE MSEs for each model
  mean_mse <- rowMeans(relative_mse, na.rm = TRUE)
  std_dev_mse <- apply(relative_mse, 1, sd, na.rm = TRUE)
  
  # Calculate mean and standard deviation of the RANKS for each model
  avg_rank <- rowMeans(ranks, na.rm = TRUE)
  std_dev_rank <- apply(ranks, 1, sd, na.rm = TRUE)
  
  # --- e. COMBINE EVERYTHING INTO THE FINAL TABLE ---
  final_results <- data.frame(
    Model = model_names,
    relative_mse,
    mean = mean_mse,
    std_dev = std_dev_mse,
    avg_rank = avg_rank,
    std_rank = std_dev_rank
  )
  
  # Set the first column name to match the input's first column name
  names(final_results)[1] <- names(input_df)[1]
  
  # --- f. ROUND ALL NUMERIC COLUMNS TO 2 DECIMALS ---
  numeric_cols <- sapply(final_results, is.numeric)
  final_results[numeric_cols] <- round(final_results[numeric_cols], 2)
  
  return(final_results)
}


# --- 2. MAIN SCRIPT LOGIC ---
input_file <- "C:/Users/rowan/Documents/finalist_results_bachelor_thesis.xlsx" # <-- Change path here

# Check if the file exists before trying to read it
if (!file.exists(input_file)) {
  stop("Error: Input file '", input_file, "' not found. Make sure it's in the working directory.")
}

# Read the full dataset from the CSV file
full_data <- read_excel(input_file)

# --- Process Replication Data (First 5 models) ---
message("Processing replication data (first 5 models)...")
replication_data <- full_data[1:5, ]
replication_output <- process_mse_data(replication_data)

# Save the replication results to a new CSV file
write.csv(replication_output, "replication_results.csv", row.names = FALSE)
message("Saved 'replication_results.csv'")

# --- Process Total Data (All 33 models) ---
message("\nProcessing total data (all models)...")
total_output <- process_mse_data(full_data)

# Save the total results to a new CSV file
write.csv(total_output, "total_results_new.csv", row.names = FALSE)
message("Saved 'total_results_new.csv'")

# --- Process Original Data from the Paper ---
# Read prepared file --- Make sure it is prepared correctly
orig_file <- "C:/Users/rowan/Documents/pilot_paper_results.xlsx" # <-- Change path here
orig_data <- read_excel(orig_file)

message("\nProcessing original paper's data")
orig_output <- process_mse_data(orig_data)

# Save the original paper's results to a new CSV file
write.csv(orig_output, "orig_paper_results.csv", row.names = FALSE)
message("Saved 'total_results.csv'")


# --- 3. SCRIPT FINISHED ---
message("\nProcessing complete.")

