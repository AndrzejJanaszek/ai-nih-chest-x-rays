"""
GUI Application for DenseNet-121 Chest X-ray Classification.
Allows users to upload X-ray images and get diagnostic predictions.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import os
import sys
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEVICE, ALL_LABELS, NUM_CLASSES, PHASE2_CHECKPOINTS,
    print_device_info, DISEASE_THRESHOLDS
)
from model import create_densenet121_model
from checkpoint import load_checkpoint
from transforms import get_val_transforms


class ChestXrayDiagnosisGUI:
    """GUI Application for X-ray Diagnosis"""
    
    def __init__(self, root, checkpoint_epoch=20):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window
            checkpoint_epoch: Which epoch checkpoint to load (default: 20)
        """
        self.root = root
        self.root.title("Chest X-ray Classification System")
        self.root.geometry("900x1000")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.model = None
        self.transform = None
        self.image_path = None
        self.current_image = None
        self.checkpoint_epoch = checkpoint_epoch
        self.csv_data = None
        
        # Load CSV data
        self.load_csv_data()
        
        # Load model
        print(f"\n[Loading Model] Checkpoint epoch: {self.checkpoint_epoch}")
        self.load_model()
        
        # Build GUI
        self.build_gui()
        
        print("✓ GUI ready")
    
    def load_csv_data(self):
        """Load CSV data with ground truth labels"""
        try:
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data',
                'Data_Entry_2017.csv'
            )
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            print(f"Loading CSV data from: {csv_path}")
            self.csv_data = pd.read_csv(csv_path)
            print(f"✓ CSV data loaded ({len(self.csv_data)} records)")
            
        except Exception as e:
            print(f"✗ Warning: Could not load CSV data: {e}")
            print("  Ground truth labels will not be available")
            self.csv_data = None
    
    def get_ground_truth_labels(self, image_name):
        """
        Get ground truth labels from CSV for the given image name.
        
        Args:
            image_name: The image file name (e.g., "00000001_000.png")
            
        Returns:
            List of ground truth labels, or empty list if not found
        """
        if self.csv_data is None:
            return []
        
        try:
            # Remove any directory path and get just the filename
            image_name = os.path.basename(image_name)
            
            # Find the record in CSV
            record = self.csv_data[self.csv_data['Image Index'] == image_name]
            
            if record.empty:
                return []
            
            # Get the Finding Labels and split by '|'
            labels_str = record.iloc[0]['Finding Labels']
            labels = [label.strip() for label in str(labels_str).split('|')]
            
            return labels
        
        except Exception as e:
            print(f"Error getting ground truth labels: {e}")
            return []
    
    def load_model(self):
        """Load model from checkpoint"""
        try:
            print_device_info()
            
            # Create model
            print(f"Creating DenseNet-121 model...")
            self.model = create_densenet121_model(DEVICE, NUM_CLASSES)
            self.model.eval()
            
            # Load checkpoint
            checkpoint_path = os.path.join(
                PHASE2_CHECKPOINTS, 
                f'checkpoint_epoch_{self.checkpoint_epoch}.pt'
            )
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Available checkpoints should be in: {PHASE2_CHECKPOINTS}"
                )
            
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.model, _, _, epoch, loss = load_checkpoint(
                checkpoint_path, 
                self.model, 
                optimizer=None, 
                scheduler=None, 
                device=DEVICE
            )
            self.model.eval()
            print(f"✓ Model loaded successfully (epoch {epoch}, loss: {loss:.4f})")
            
            # Get transformation pipeline
            self.transform = get_val_transforms()
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n{e}")
            raise
    
    def build_gui(self):
        """Build the GUI layout"""
        # Create Canvas and Scrollbar for main application
        canvas = tk.Canvas(self.root, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=canvas.yview)
        
        # Main frame that will be inside canvas
        main_frame = tk.Frame(canvas, bg='white')
        
        # Create window in canvas
        window_id = canvas.create_window((0, 0), window=main_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Update canvas scrollregion and window width
        def on_frame_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Set canvas window width to match canvas width
            canvas.itemconfig(window_id, width=canvas.winfo_width())
        
        main_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_frame_configure)
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Continue with regular frame packing inside main_frame
        main_frame_content = tk.Frame(main_frame, bg='white')
        main_frame_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ============================================================
        # TITLE
        # ============================================================
        title_frame = tk.Frame(main_frame_content, bg='#2c3e50')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(
            title_frame,
            text="Chest X-ray Diagnostic System",
            font=("Arial", 18, "bold"),
            bg='#2c3e50',
            fg='white',
            pady=10
        )
        title_label.pack()
        
        # ============================================================
        # FILE SELECTION SECTION
        # ============================================================
        file_frame = tk.LabelFrame(
            main_frame_content, 
            text="Step 1: Load X-ray Image", 
            font=("Arial", 11, "bold"),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        file_frame.pack(fill=tk.X, pady=10)
        
        button_frame = tk.Frame(file_frame, bg='#ecf0f1')
        button_frame.pack(fill=tk.X)
        
        self.upload_btn = tk.Button(
            button_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Arial", 10, "bold"),
            bg='#3498db',
            fg='white',
            padx=15,
            pady=8,
            cursor="hand2"
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑 Clear",
            command=self.clear_image,
            font=("Arial", 10),
            bg='#95a5a6',
            fg='white',
            padx=10,
            pady=8,
            cursor="hand2"
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # File path display
        self.path_label = tk.Label(
            file_frame,
            text="No image loaded",
            font=("Arial", 9),
            bg='#ecf0f1',
            fg='#7f8c8d',
            wraplength=700,
            justify=tk.LEFT
        )
        self.path_label.pack(fill=tk.X, pady=(10, 0))
        
        # ============================================================
        # IMAGE PREVIEW SECTION
        # ============================================================
        image_frame = tk.LabelFrame(
            main_frame_content,
            text="Step 2: Image Preview",
            font=("Arial", 11, "bold"),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        image_frame.pack(fill=tk.X, pady=10)
        
        # Use Canvas for better image display
        self.image_canvas = tk.Canvas(
            image_frame,
            width=300,
            height=300,
            bg='#bdc3c7',
            highlightthickness=0
        )
        self.image_canvas.pack(padx=10, pady=10)
        self.image_label = None  # Placeholder for compatibility
        
        # ============================================================
        # DIAGNOSIS SECTION
        # ============================================================
        diagnosis_frame = tk.LabelFrame(
            main_frame_content,
            text="Step 3: Run Diagnosis",
            font=("Arial", 11, "bold"),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        diagnosis_frame.pack(fill=tk.X, pady=10)
        
        self.diagnose_btn = tk.Button(
            diagnosis_frame,
            text="🔍 Analyze Image",
            command=self.diagnose,
            font=("Arial", 11, "bold"),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.diagnose_btn.pack(pady=5)
        
        # ============================================================
        # RESULTS SECTION
        # ============================================================
        results_frame = tk.LabelFrame(
            main_frame_content,
            text="Step 4: Diagnostic Results",
            font=("Arial", 11, "bold"),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create canvas with scrollbar for results
        results_canvas = tk.Canvas(results_frame, bg='white', highlightthickness=0)
        results_scrollbar = tk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_canvas.yview)
        scrollable_frame = tk.Frame(results_canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        )
        
        results_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_container = scrollable_frame
        
        self.results_label = tk.Label(
            self.results_container,
            text="Results will appear here after analysis",
            font=("Arial", 10),
            bg='white',
            fg='#7f8c8d',
            justify=tk.LEFT,
            wraplength=700
        )
        self.results_label.pack(pady=10)
        
        # ============================================================
        # BOTTOM BUTTONS
        # ============================================================
        bottom_frame = tk.Frame(main_frame_content, bg='white')
        bottom_frame.pack(fill=tk.X, pady=10)
        
        exit_btn = tk.Button(
            bottom_frame,
            text="❌ Exit",
            command=self.root.quit,
            font=("Arial", 10, "bold"),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=8,
            cursor="hand2"
        )
        exit_btn.pack(side=tk.RIGHT, padx=5)
        
        info_label = tk.Label(
            bottom_frame,
            text=f"Model: DenseNet-121 (Epoch {self.checkpoint_epoch}) | Device: {DEVICE}",
            font=("Arial", 8),
            bg='white',
            fg='#7f8c8d'
        )
        info_label.pack(side=tk.LEFT, padx=5)
    
    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Chest X-ray Image",
            filetypes=file_types
        )
        
        if file_path:
            self.image_path = file_path
            self.load_and_display_image()
            self.diagnose_btn.config(state=tk.NORMAL)
    
    def load_and_display_image(self):
        """Load and display the uploaded image"""
        try:
            # Load original image
            self.current_image = Image.open(self.image_path).convert('RGB')
            
            # Scale down if image is larger than 256px on any dimension
            max_dim = max(self.current_image.size)
            if max_dim > 256:
                # Maintain aspect ratio while scaling to fit in 256x256
                self.current_image.thumbnail((256, 256), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(self.current_image)
            
            # Display on Canvas
            self.image_canvas.delete("all")
            self.image_canvas.create_image(150, 150, image=photo)
            self.image_canvas.image = photo  # Keep a reference
            
            # Update path label
            file_name = os.path.basename(self.image_path)
            file_size = os.path.getsize(self.image_path) / 1024  # KB
            self.path_label.config(
                text=f"📄 File: {file_name}\n📊 Size: {file_size:.1f} KB\n📐 Dimensions: {self.current_image.size[0]}x{self.current_image.size[1]}",
                fg='#27ae60'
            )
            
            # Clear previous results
            self.clear_results()
            
        except Exception as e:
            messagebox.showerror("Image Loading Error", f"Failed to load image:\n{e}")
    
    def diagnose(self):
        """Run diagnosis on the loaded image"""
        if self.current_image is None or self.model is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            # Update button state
            self.diagnose_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
            self.root.update()
            
            # Prepare image for model
            image_tensor = self.transform(self.current_image)
            image_batch = image_tensor.unsqueeze(0).to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_batch)
                predictions = torch.sigmoid(outputs)
            
            # Get predictions
            pred_values = predictions[0].cpu().numpy()
            
            # Get ground truth labels from CSV
            ground_truth_labels = self.get_ground_truth_labels(self.image_path)
            
            # Display results
            self.display_results(pred_values, ground_truth_labels)
            
            # Restore button
            self.diagnose_btn.config(state=tk.NORMAL, text="🔍 Analyze Image")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze image:\n{e}")
            self.diagnose_btn.config(state=tk.NORMAL, text="🔍 Analyze Image")
    
    def display_results(self, predictions, ground_truth_labels=None):
        """Display diagnostic results with predictions and ground truth labels"""
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        # Add results header
        header = tk.Label(
            self.results_container,
            text="Diagnostic Results",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#2c3e50'
        )
        header.pack(pady=(10, 5), padx=10, anchor=tk.W)
        
        # Display ground truth labels if available
        if ground_truth_labels:
            gt_frame = tk.Frame(self.results_container, bg='#e8f8f5', relief=tk.SOLID, borderwidth=1)
            gt_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
            
            gt_title = tk.Label(
                gt_frame,
                text="✓ Ground Truth (CSV Labels):",
                font=("Arial", 10, "bold"),
                bg='#e8f8f5',
                fg='#16a085'
            )
            gt_title.pack(anchor=tk.W, padx=10, pady=(8, 2))
            
            gt_text = tk.Label(
                gt_frame,
                text=" | ".join(ground_truth_labels),
                font=("Arial", 10),
                bg='#e8f8f5',
                fg='#27ae60',
                wraplength=600,
                justify=tk.LEFT
            )
            gt_text.pack(anchor=tk.W, padx=10, pady=(2, 8))
        
        # ============================================================
        # POSITIVE FINDINGS (above threshold)
        # ============================================================
        positive_findings = []
        for idx, confidence in enumerate(predictions):
            label_name = ALL_LABELS[idx]
            threshold = DISEASE_THRESHOLDS.get(label_name, 0.5)
            if confidence >= threshold:
                positive_findings.append((label_name, confidence, threshold, idx))
        
        # Sort positive findings by confidence (descending)
        positive_findings.sort(key=lambda x: x[1], reverse=True)
        
        if positive_findings:
            # Positive findings header
            pos_header = tk.Label(
                self.results_container,
                text="🔴 POSITIVE FINDINGS (Above Threshold):",
                font=("Arial", 11, "bold"),
                bg='white',
                fg='#c0392b'
            )
            pos_header.pack(pady=(10, 5), padx=10, anchor=tk.W)
            
            # Display positive findings
            for idx, (label_name, confidence, threshold, _) in enumerate(positive_findings, 1):
                is_in_ground_truth = (ground_truth_labels and label_name in ground_truth_labels)
                
                # Create frame for each positive finding
                result_frame = tk.Frame(self.results_container, bg='#fadbd8', relief=tk.SOLID, borderwidth=1)
                result_frame.pack(fill=tk.X, padx=10, pady=5)
                
                # Left side: Label name and rank
                left_frame = tk.Frame(result_frame, bg='#fadbd8')
                left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)
                
                label_text_prefix = "✓ " if is_in_ground_truth else "● "
                label_text = tk.Label(
                    left_frame,
                    text=f"{label_text_prefix}{idx}. {label_name}",
                    font=("Arial", 10, "bold"),
                    bg='#fadbd8',
                    fg='#922b21' if is_in_ground_truth else '#c0392b',
                    anchor=tk.W,
                    justify=tk.LEFT
                )
                label_text.pack(anchor=tk.W)
                
                # Threshold info
                threshold_text = tk.Label(
                    left_frame,
                    text=f"Threshold: {threshold:.2f}",
                    font=("Arial", 8),
                    bg='#fadbd8',
                    fg='#7f8c8d',
                    anchor=tk.W,
                    justify=tk.LEFT
                )
                threshold_text.pack(anchor=tk.W)
                
                # Right side: Confidence percentage
                confidence_text = tk.Label(
                    result_frame,
                    text=f"{confidence*100:.1f}%",
                    font=("Arial", 11, "bold"),
                    bg='#fadbd8',
                    fg='#922b21',
                    anchor=tk.E
                )
                confidence_text.pack(side=tk.RIGHT, padx=10, pady=5)
                
                # Progress bar
                bar_frame = tk.Frame(result_frame, bg='white', height=20)
                bar_frame.pack(fill=tk.X, padx=10, pady=5)
                
                bar_color = '#922b21' if is_in_ground_truth else '#e74c3c'
                bar_width = int(confidence * 200)  # Scale to max 200px
                bar = tk.Canvas(
                    bar_frame,
                    width=bar_width,
                    height=15,
                    bg=bar_color,
                    highlightthickness=0
                )
                bar.pack(anchor=tk.W)
        
        # ============================================================
        # NEGATIVE FINDINGS (below threshold)
        # ============================================================
        negative_findings = []
        for idx, confidence in enumerate(predictions):
            label_name = ALL_LABELS[idx]
            threshold = DISEASE_THRESHOLDS.get(label_name, 0.5)
            if confidence < threshold:
                negative_findings.append((label_name, confidence, threshold, idx))
        
        # Sort negative findings by confidence (descending)
        negative_findings.sort(key=lambda x: x[1], reverse=True)
        
        if negative_findings:
            # Negative findings header
            neg_header = tk.Label(
                self.results_container,
                text="🟢 NEGATIVE FINDINGS (Below Threshold):",
                font=("Arial", 11, "bold"),
                bg='white',
                fg='#27ae60'
            )
            neg_header.pack(pady=(15, 5), padx=10, anchor=tk.W)
            
            # Display negative findings
            for idx, (label_name, confidence, threshold, _) in enumerate(negative_findings, 1):
                is_in_ground_truth = (ground_truth_labels and label_name in ground_truth_labels)
                
                # Create frame for each negative finding
                result_frame = tk.Frame(self.results_container, bg='#d5f4e6', relief=tk.FLAT)
                result_frame.pack(fill=tk.X, padx=10, pady=3)
                
                # Left side: Label name
                label_text = tk.Label(
                    result_frame,
                    text=f"{label_name}",
                    font=("Arial", 9),
                    bg='#d5f4e6',
                    fg='#27ae60',
                    width=28,
                    anchor=tk.W
                )
                label_text.pack(side=tk.LEFT, padx=10, pady=3)
                
                # Middle: Threshold info
                threshold_text = tk.Label(
                    result_frame,
                    text=f"Threshold: {threshold:.2f}",
                    font=("Arial", 8),
                    bg='#d5f4e6',
                    fg='#7f8c8d'
                )
                threshold_text.pack(side=tk.LEFT, padx=5)
                
                # Right side: Confidence percentage
                confidence_text = tk.Label(
                    result_frame,
                    text=f"{confidence*100:.1f}%",
                    font=("Arial", 9),
                    bg='#d5f4e6',
                    fg='#27ae60',
                    anchor=tk.E
                )
                confidence_text.pack(side=tk.RIGHT, padx=10, pady=3)
        
        # Add instructions for next step
        instruction_label = tk.Label(
            self.results_container,
            text="\n👉 You can upload another image or exit the application",
            font=("Arial", 9, "italic"),
            bg='white',
            fg='#7f8c8d'
        )
        instruction_label.pack(pady=10)

    
    def clear_image(self):
        """Clear the current image"""
        self.image_path = None
        self.current_image = None
        self.image_canvas.delete("all")
        self.image_canvas.image = None
        self.path_label.config(text="No image loaded", fg='#7f8c8d')
        self.diagnose_btn.config(state=tk.DISABLED)
        self.clear_results()
    
    def clear_results(self):
        """Clear the results display"""
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        self.results_label = tk.Label(
            self.results_container,
            text="Results will appear here after analysis",
            font=("Arial", 10),
            bg='white',
            fg='#7f8c8d',
            justify=tk.LEFT,
            wraplength=700
        )
        self.results_label.pack(pady=10)


def main(checkpoint_epoch=20):
    """
    Main entry point for the GUI application.
    
    Args:
        checkpoint_epoch: Which epoch checkpoint to load (default: 20)
    """
    print("\n" + "=" * 60)
    print("Chest X-ray Classification GUI")
    print("=" * 60)
    
    root = tk.Tk()
    app = ChestXrayDiagnosisGUI(root, checkpoint_epoch=checkpoint_epoch)
    root.mainloop()


if __name__ == "__main__":
    # You can specify which epoch to load:
    # main(checkpoint_epoch=20)  # Load epoch 20 checkpoint
    # main(checkpoint_epoch=10)  # Load epoch 10 checkpoint
    
    main(checkpoint_epoch=20)  # Default to epoch 20
