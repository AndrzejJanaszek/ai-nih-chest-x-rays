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
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEVICE, ALL_LABELS, NUM_CLASSES, PHASE2_CHECKPOINTS,
    print_device_info
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
        
        # Load model
        print(f"\n[Loading Model] Checkpoint epoch: {self.checkpoint_epoch}")
        self.load_model()
        
        # Build GUI
        self.build_gui()
        
        print("✓ GUI ready")
    
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
        # Main frame with scrollbar support
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ============================================================
        # TITLE
        # ============================================================
        title_frame = tk.Frame(main_frame, bg='#2c3e50')
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
            main_frame, 
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
            main_frame,
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
            main_frame,
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
            main_frame,
            text="Step 4: Diagnostic Results",
            font=("Arial", 11, "bold"),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create canvas with scrollbar for results
        canvas = tk.Canvas(results_frame, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(results_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
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
        bottom_frame = tk.Frame(main_frame, bg='white')
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
            
            # Display results
            self.display_results(pred_values)
            
            # Restore button
            self.diagnose_btn.config(state=tk.NORMAL, text="🔍 Analyze Image")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze image:\n{e}")
            self.diagnose_btn.config(state=tk.NORMAL, text="🔍 Analyze Image")
    
    def display_results(self, predictions):
        """Display diagnostic results"""
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
        
        # Sort predictions by confidence (descending)
        sorted_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)
        
        # Display each diagnosis
        for rank, idx in enumerate(sorted_indices, 1):
            label_name = ALL_LABELS[idx]
            confidence = predictions[idx]
            
            # Create frame for each result
            result_frame = tk.Frame(self.results_container, bg='#f0f0f0', relief=tk.FLAT)
            result_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Label name and rank
            label_text = tk.Label(
                result_frame,
                text=f"{rank}. {label_name}",
                font=("Arial", 10, "bold"),
                bg='#f0f0f0',
                fg='#2c3e50',
                width=25,
                anchor=tk.W
            )
            label_text.pack(side=tk.LEFT, padx=10, pady=5)
            
            # Confidence percentage
            percentage = confidence * 100
            confidence_text = tk.Label(
                result_frame,
                text=f"{percentage:.1f}%",
                font=("Arial", 10, "bold"),
                bg='#f0f0f0',
                fg='#2980b9',
                width=10,
                anchor=tk.E
            )
            confidence_text.pack(side=tk.RIGHT, padx=10, pady=5)
            
            # Progress bar (visual representation)
            bar_frame = tk.Frame(result_frame, bg='white', height=20)
            bar_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Determine color based on confidence
            if confidence >= 0.7:
                bar_color = '#e74c3c'  # Red - high confidence
            elif confidence >= 0.4:
                bar_color = '#f39c12'  # Orange - medium confidence
            else:
                bar_color = '#27ae60'  # Green - low confidence
            
            bar_width = int(percentage * 2)  # Scale to max 200px
            bar = tk.Canvas(
                bar_frame,
                width=bar_width,
                height=15,
                bg=bar_color,
                highlightthickness=0
            )
            bar.pack(anchor=tk.W)
        
        # Add summary section
        summary_frame = tk.Frame(self.results_container, bg='white', relief=tk.SOLID, borderwidth=1)
        summary_frame.pack(fill=tk.X, padx=10, pady=(15, 5))
        
        max_idx = sorted_indices[0]
        max_confidence = predictions[max_idx]
        max_label = ALL_LABELS[max_idx]
        
        summary_text = f"⭐ Primary Finding: {max_label} ({max_confidence*100:.1f}% confidence)"
        summary_label = tk.Label(
            summary_frame,
            text=summary_text,
            font=("Arial", 11, "bold"),
            bg='white',
            fg='#e74c3c' if max_confidence >= 0.7 else '#f39c12',
            wraplength=600,
            justify=tk.LEFT,
            pady=10,
            padx=10
        )
        summary_label.pack(fill=tk.BOTH)
        
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
