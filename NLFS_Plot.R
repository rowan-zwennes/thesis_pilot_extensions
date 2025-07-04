# --- 1. Load Required Libraries ---
# Make sure you have these installed.
library(readxl)
library(ggplot2)
library(tidyr)
library(dplyr)
library(scales)

# --- 2. Define File Path and Read Data ---
file_path <- "C:/Users/rowan/Documents/final_results_nlfs_times.xlsx" # <-- Make sure path is correct

# Read the data
original_data <- read_excel(file_path)

# --- 3. Reshape Data from Wide to Long Format ---
data_long <- pivot_longer(
  original_data,
  cols = -Dimension,
  names_to = "Model",
  values_to = "Fitting_Time"
)

# --- 4. Calculate Slopes for Each Model ---
slope_data <- data_long %>%
  group_by(Model) %>%
  do(model = lm(log10(Fitting_Time) ~ log10(Dimension), data = .)) %>%
  summarise(Model = Model, Slope = coef(model)[2]) %>%
  ungroup()

# Display the calculated slopes
print("Calculated Slopes (Empirical Complexity Exponents):")
print(slope_data)

# --- 5. Create New Labels for the Legend ---
data_long_with_labels <- data_long %>%
  left_join(slope_data, by = "Model") %>%
  mutate(Legend_Label = paste0(Model, " (m = ", round(Slope, 2), ")"))

# --- 6. Generate the Plot with Corrected Text Sizes ---
publication_plot_final <- ggplot(data_long_with_labels, aes(x = Dimension, y = Fitting_Time, color = Legend_Label, shape = Legend_Label)) +
  
  geom_point(size = 3.5, alpha = 0.7) +
  
  geom_smooth(method = "lm", se = FALSE, linewidth = 1) +
  
  scale_x_log10(
    breaks = trans_breaks("log10", function(x) 10^x, n = 3),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_y_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  
  labs(
    x = "Dataset Dimension (p)",
    y = "Fitting Time in Seconds",
    color = "Algorithm:",
    shape = "Algorithm:"
  ) +
  
  theme_bw() + # Start with a clean theme
  
  # === AESTHETIC ADJUSTMENTS ===
  theme(
    legend.position = "bottom",
    axis.title = element_text(size = 30),
    legend.title = element_text(size = 30, face = "bold"),
    legend.text = element_text(size = 30),
    
    axis.text = element_text(size = 26), # Increased from default
    
    # Minor cleanup
    panel.grid.minor = element_blank()
  ) +
  
  # Arrange the legend into 2 rows to prevent it from going off the page
  guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
  
  # Use a color palette with good contrast
  scale_color_brewer(palette = "Set1")

# --- 7. Display the Plot ---
print(publication_plot_final)

# --- 8. Save the Plot to a File ---
# Saving with a slightly wider aspect ratio to better accommodate the legend
ggsave("fitting_time_vs_dimension_final.png", plot = publication_plot_final, width = 12, height = 9, dpi = 300)
