#!/usr/bin/env python3
"""
ID Card PDF Maker — improved manual crop on ORIGINAL + padding + crop marks

Dependencies:
    pip install ttkbootstrap pillow opencv-python reportlab
"""
import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ----------------------------
# Geometry / helpers
# ----------------------------
PT_PER_INCH = 72.0
MM_PER_INCH = 25.4


def mm_to_pt(mm: float) -> float:
    return mm * PT_PER_INCH / MM_PER_INCH


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = distance(br, bl)
    widthB = distance(tr, tl)
    maxWidth = int(max(widthA, widthB))
    heightA = distance(tr, br)
    heightB = distance(tl, bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def autocrop_border(pil_img, bg_tol=250):
    arr = np.array(pil_img.convert("L"))
    mask = arr < bg_tol
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pil_img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return pil_img.crop((x0, y0, x1, y1))


# ----------------------------
# Card detection
# ----------------------------
def detect_card_quad(image_bgr, use_rescale=True, min_area_ratio=0.1):
    """Return 4-point float32 quad in image coordinates or None."""
    h, w = image_bgr.shape[:2]
    image_area = h * w
    scale = 1.0
    small = image_bgr
    if use_rescale:
        scale = 1000.0 / max(h, w)
        if scale < 1:
            small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        else:
            small = image_bgr.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best = None
    best_score = 0
    
    for c in contours[:15]:
        area = cv2.contourArea(c)
        # Area should be at least min_area_ratio of the image
        if area < (image_area * min_area_ratio * scale * scale):
            continue
            
        peri = cv2.arcLength(c, True)
        # Try different approximation tolerances for better quad detection
        for epsilon in [0.015, 0.02, 0.025, 0.03]:
            approx = cv2.approxPolyDP(c, epsilon * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(-1, 2).astype("float32")
                ordered = order_points(pts)
                (tl, tr, br, bl) = ordered
                
                # Calculate more robust dimensions
                wA = distance(br, bl)
                wB = distance(tr, tl)
                hA = distance(tr, br)
                hB = distance(tl, bl)
                w_est = (wA + wB) / 2.0  # Average width
                h_est = (hA + hB) / 2.0  # Average height
                
                if h_est == 0 or w_est == 0:
                    continue
                    
                ratio = max(w_est / h_est, h_est / w_est)
                # Accept plausible card shapes (ID cards are typically 1.4 to 1.8 ratio)
                if 1.2 <= ratio <= 2.2:
                    # Score based on area and aspect ratio closeness to standard ID card
                    standard_ratio = 85.6 / 54.0  # Standard ID card ratio
                    ratio_score = 1.0 / (1.0 + abs(ratio - standard_ratio))
                    area_score = area / (image_area * scale * scale)
                    combined_score = ratio_score * 0.7 + area_score * 0.3
                    
                    if combined_score > best_score:
                        best = pts
                        best_score = combined_score
                break

    if best is None:
        return None
    # map back to original scale if resized
    if use_rescale and scale < 1:
        best = best / scale
    return best.astype("float32")


# ----------------------------
# Crop card function with manual-box-aware detection
# ----------------------------
def crop_card_with_optional_manual(original_path, manual_box=None, enforce_ratio=False, ratio_w=85.6, ratio_h=54.0, enhance_image=True):
    """
    original_path: path to original image
    manual_box: (x0,y0,x1,y1) in original image pixel coords, or None
    enforce_ratio: crop result to exact ratio (center-crop after warp)
    """
    img = cv2.imread(original_path)
    if img is None:
        raise ValueError(f"Cannot open {original_path}")

    # If manual_box provided, attempt to detect a quad within the manual subimage
    if manual_box:
        x0, y0, x1, y1 = manual_box
        x0, x1 = max(0, int(x0)), min(img.shape[1], int(x1))
        y0, y1 = max(0, int(y0)), min(img.shape[0], int(y1))
        sub = img[y0:y1, x0:x1]
        q = detect_card_quad(sub, use_rescale=True)
        if q is not None:
            # map sub-quad back to original image coords
            q[:, 0] += x0
            q[:, 1] += y0
            warped = four_point_transform(img, q)
        else:
            # no quad found in sub-region — fallback to axis-aligned crop of manual box
            cropped = img[y0:y1, x0:x1]
            warped = cropped
    else:
        # no manual box: try full-image detection first
        # Try different detection parameters for better results
        q = detect_card_quad(img, use_rescale=True, min_area_ratio=0.05)
        if q is None:
            q = detect_card_quad(img, use_rescale=True, min_area_ratio=0.02)
        
        if q is not None:
            warped = four_point_transform(img, q)
        else:
            # Enhanced fallback with better preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            v = np.median(blur)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(blur, lower, upper)
            
            # Morphological operations to connect broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError("Could not detect the card automatically. Please use manual crop or try a different image.")
            
            # Find the largest rectangular-ish contour
            best_contour = None
            best_score = 0
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
                area = cv2.contourArea(c)
                if area < img.shape[0] * img.shape[1] * 0.02:  # Must be at least 2% of image
                    continue
                    
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                w, h = rect[1]
                if h == 0: continue
                
                aspect_ratio = max(w/h, h/w)
                if 1.2 <= aspect_ratio <= 2.2:  # Reasonable card aspect ratio
                    score = area / (aspect_ratio * 0.1 + 1)  # Prefer larger areas with good ratios
                    if score > best_score:
                        best_score = score
                        best_contour = box
                        
            if best_contour is None:
                raise ValueError("Could not find a suitable card shape. Please use manual crop.")
                
            warped = four_point_transform(img, best_contour.astype("float32"))

    pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    pil_img = autocrop_border(pil_img, bg_tol=245)  # More aggressive border removal
    
    # Optional image enhancement
    if enhance_image:
        # Slight contrast and sharpness enhancement
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.05)

    if enforce_ratio:
        # Center-crop to exact ratio (mm-based ratio)
        target_ratio = ratio_w / ratio_h
        w, h = pil_img.size
        cur = w / h
        if cur > target_ratio:
            new_w = int(h * target_ratio)
            x0 = (w - new_w) // 2
            pil_img = pil_img.crop((x0, 0, x0 + new_w, h))
        else:
            new_h = int(w / target_ratio)
            y0 = (h - new_h) // 2
            pil_img = pil_img.crop((0, y0, w, y0 + new_h))
    return pil_img


# ----------------------------
# PDF layout (with padding)
# ----------------------------
def layout_counts(page_w, page_h, card_w_pt, card_h_pt, margin_pt, spacing_pt):
    usable_w = page_w - 2 * margin_pt
    usable_h = page_h - 2 * margin_pt
    # number of columns and rows given spacing between cards
    cols = max(1, int((usable_w + spacing_pt) // (card_w_pt + spacing_pt)))
    rows = max(1, int((usable_h + spacing_pt) // (card_h_pt + spacing_pt)))
    return rows, cols


def draw_tiled_pdf(output_pdf, img_path, card_w_pt, card_h_pt, margin_mm=5.0, spacing_mm=5.0, orientation="portrait", crop_marks=False, crop_mark_offset_mm=2.0):
    page_w, page_h = A4
    if orientation == "landscape":
        page_w, page_h = page_h, page_w

    margin_pt = mm_to_pt(margin_mm)
    spacing_pt = mm_to_pt(spacing_mm)
    rows, cols = layout_counts(page_w, page_h, card_w_pt, card_h_pt, margin_pt, spacing_pt)

    total_w = cols * card_w_pt + (cols - 1) * spacing_pt
    total_h = rows * card_h_pt + (rows - 1) * spacing_pt
    start_x = (page_w - total_w) / 2
    start_y = (page_h - total_h) / 2

    c = canvas.Canvas(output_pdf, pagesize=(page_w, page_h))
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.setLineWidth(0.5)
    
    for r in range(rows):
        for col in range(cols):
            x = start_x + col * (card_w_pt + spacing_pt)
            y = start_y + (rows - 1 - r) * (card_h_pt + spacing_pt)
            c.drawImage(img_path, x, y, width=card_w_pt, height=card_h_pt)

    if crop_marks:
        mark_length = mm_to_pt(4.0)
        mark_offset = mm_to_pt(crop_mark_offset_mm)
        c.setStrokeColorRGB(0.0, 0.0, 0.0)
        c.setLineWidth(0.25)
        
        for r in range(rows):
            for col in range(cols):
                x0 = start_x + col * (card_w_pt + spacing_pt)
                y0 = start_y + (rows - 1 - r) * (card_h_pt + spacing_pt)
                x1 = x0 + card_w_pt
                y1 = y0 + card_h_pt
                
                # Top-left corner marks (offset outward from card)
                c.line(x0 - mark_offset - mark_length, y0 - mark_offset, x0 - mark_offset, y0 - mark_offset)
                c.line(x0 - mark_offset, y0 - mark_offset - mark_length, x0 - mark_offset, y0 - mark_offset)
                
                # Top-right corner marks
                c.line(x1 + mark_offset, y0 - mark_offset, x1 + mark_offset + mark_length, y0 - mark_offset)
                c.line(x1 + mark_offset, y0 - mark_offset - mark_length, x1 + mark_offset, y0 - mark_offset)
                
                # Bottom-left corner marks
                c.line(x0 - mark_offset - mark_length, y1 + mark_offset, x0 - mark_offset, y1 + mark_offset)
                c.line(x0 - mark_offset, y1 + mark_offset, x0 - mark_offset, y1 + mark_offset + mark_length)
                
                # Bottom-right corner marks
                c.line(x1 + mark_offset, y1 + mark_offset, x1 + mark_offset + mark_length, y1 + mark_offset)
                c.line(x1 + mark_offset, y1 + mark_offset, x1 + mark_offset, y1 + mark_offset + mark_length)

    c.showPage()
    c.save()


# ----------------------------
# Simple ROI Editor (axis aligned)
# ----------------------------
class RoiEditor(tk.Toplevel):
    def __init__(self, master, pil_image, title="Adjust Crop"):
        super().__init__(master)
        self.title(title)
        self.resizable(True, True)
        self.original = pil_image
        self.crop_box = None  # (x0, y0, x1, y1) in original image coords
        
        # Make window modal
        self.transient(master)
        self.grab_set()

        # Instructions
        inst_frame = ttk.Frame(self, padding=10)
        inst_frame.pack(fill=X)
        ttk.Label(inst_frame, text="📋 Instructions: Click and drag to select the card area for cropping", 
                 font=("Helvetica", 10)).pack()

        self.canvas = tk.Canvas(self, background="#2c2c2c")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

        btns = ttk.Frame(self, padding=10)
        btns.pack(fill=X)
        ttk.Label(btns, text="💡 Double-click to auto-detect card bounds", 
                 font=("Helvetica", 9), foreground="#666").pack(side=LEFT)
        ttk.Button(btns, text="🔄 Reset", bootstyle=SECONDARY, command=self._reset).pack(side=RIGHT, padx=3)
        ttk.Button(btns, text="✅ Apply", bootstyle=SUCCESS, command=self._apply).pack(side=RIGHT, padx=3)
        ttk.Button(btns, text="❌ Cancel", bootstyle=WARNING, command=self._cancel).pack(side=RIGHT, padx=3)

        self._drag_start = None
        self._rect_id = None
        self._scale = 1.0
        self._offset = (0, 0)
        self.preview_imgtk = None
        self.minsize(800, 600)
        
        # Center the window
        self.geometry("900x700")
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.winfo_screenheight() // 2) - (700 // 2)
        self.geometry(f"900x700+{x}+{y}")
        
    def _on_double_click(self, event):
        """Auto-detect card bounds on double-click"""
        try:
            # Convert PIL to OpenCV format
            cv_img = cv2.cvtColor(np.array(self.original), cv2.COLOR_RGB2BGR)
            # Try to detect card quad
            quad = detect_card_quad(cv_img, use_rescale=False)
            if quad is not None:
                x_coords = quad[:, 0]
                y_coords = quad[:, 1]
                x0, x1 = int(min(x_coords)), int(max(x_coords))
                y0, y1 = int(min(y_coords)), int(max(y_coords))
                
                # Add small margin
                margin = 10
                iw, ih = self.original.size
                self.crop_box = (
                    max(0, x0 - margin), max(0, y0 - margin),
                    min(iw, x1 + margin), min(ih, y1 + margin)
                )
                self._on_resize()
            else:
                # Fallback: use center 80% of image
                iw, ih = self.original.size
                margin_w, margin_h = int(iw * 0.1), int(ih * 0.1)
                self.crop_box = (margin_w, margin_h, iw - margin_w, ih - margin_h)
                self._on_resize()
        except Exception:
            pass  # Ignore auto-detect errors

    def _on_resize(self, event=None):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.original.size
        scale = min(max(0.01, cw / iw), max(0.01, ch / ih))
        self._scale = scale
        nw, nh = int(iw * scale), int(ih * scale)
        resized = self.original.resize((nw, nh), Image.LANCZOS)
        self.preview_imgtk = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self._offset = ((cw - nw) // 2, (ch - nh) // 2)
        self.canvas.create_image(self._offset[0], self._offset[1], image=self.preview_imgtk, anchor="nw")
        if self.crop_box:
            x0, y0, x1, y1 = self.crop_box
            sx0 = self._offset[0] + int(x0 * self._scale)
            sy0 = self._offset[1] + int(y0 * self._scale)
            sx1 = self._offset[0] + int(x1 * self._scale)
            sy1 = self._offset[1] + int(y1 * self._scale)
            self._rect_id = self.canvas.create_rectangle(sx0, sy0, sx1, sy1, outline="#00FFAA", width=2)

    def _on_press(self, event):
        self._drag_start = (event.x, event.y)
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_drag(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        if self._rect_id:
            self.canvas.coords(self._rect_id, x0, y0, x1, y1)
        else:
            self._rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00FFAA", width=2)

    def _on_release(self, event):
        if not self._rect_id:
            return
        x0, y0, x1, y1 = self.canvas.coords(self._rect_id)
        ix0 = int((x0 - self._offset[0]) / self._scale)
        iy0 = int((y0 - self._offset[1]) / self._scale)
        ix1 = int((x1 - self._offset[0]) / self._scale)
        iy1 = int((y1 - self._offset[1]) / self._scale)
        iw, ih = self.original.size
        ix0, ix1 = sorted((max(0, ix0), min(iw, ix1)))
        iy0, iy1 = sorted((max(0, iy0), min(ih, iy1)))
        if ix1 - ix0 > 10 and iy1 - iy0 > 10:
            self.crop_box = (ix0, iy0, ix1, iy1)

    def _reset(self):
        self.crop_box = None
        self._on_resize()

    def _apply(self):
        self.destroy()

    def _cancel(self):
        self.crop_box = None
        self.destroy()


# ----------------------------
# Main application
# ----------------------------
class IDCardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ID Card PDF Maker — Professional Edition")
        self.root.geometry("1100x800")
        self.root.minsize(900, 700)

        # State
        self.front_path = None
        self.back_path = None
        self.front_preview_img = None   # PIL
        self.back_preview_img = None
        self.front_manual_box = None  # coords in original image for manual crop
        self.back_manual_box = None

        # Options
        self.enforce_ratio_var = tk.BooleanVar(value=True)
        self.rotate_back_var = tk.BooleanVar(value=False)
        self.crop_marks_var = tk.BooleanVar(value=True)
        self.padding_mm = tk.DoubleVar(value=5.0)
        self.margin_mm = tk.DoubleVar(value=5.0)
        self.card_w_mm = tk.DoubleVar(value=85.6)
        self.card_h_mm = tk.DoubleVar(value=54.0)
        self.crop_mark_offset_mm = tk.DoubleVar(value=2.0)
        
        # Status tracking
        self.status_text = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar()

        self._build_ui()

    def _build_ui(self):
        # Header with title and status
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill=X)
        
        ttk.Label(header, text="🆔 ID Card PDF Maker", font=("Helvetica", 20, "bold")).pack()
        ttk.Label(header, text="Professional bulk ID card printing solution", 
                 font=("Helvetica", 10), bootstyle=SECONDARY).pack(pady=(0, 5))
        
        # Status bar
        status_frame = ttk.Frame(header)
        status_frame.pack(fill=X, pady=5)
        ttk.Label(status_frame, text="Status:", font=("Helvetica", 9, "bold")).pack(side=LEFT)
        ttk.Label(status_frame, textvariable=self.status_text, font=("Helvetica", 9)).pack(side=LEFT, padx=(5, 0))
        
        ttk.Separator(self.root, orient=HORIZONTAL).pack(fill=X, pady=5)

        body = ttk.Frame(self.root, padding=15)
        body.pack(fill=BOTH, expand=True)

        # Image selection section
        img_section = ttk.Labelframe(body, text="📷 Image Selection", padding=15)
        img_section.pack(fill=X, pady=(0, 10))
        
        # Front image controls in a nice layout
        front_frame = ttk.Frame(img_section)
        front_frame.pack(fill=X, pady=5)
        
        front_left = ttk.Frame(front_frame)
        front_left.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(front_left, text="Front Side (Required):", font=("Helvetica", 10, "bold")).pack(anchor=W)
        self.front_name = ttk.Label(front_left, text="No file selected", foreground="#666")
        self.front_name.pack(anchor=W, pady=(2, 0))
        
        front_buttons = ttk.Frame(front_frame)
        front_buttons.pack(side=RIGHT)
        ttk.Button(front_buttons, text="📂 Browse", bootstyle=PRIMARY, command=self.load_front).pack(side=RIGHT, padx=2)
        ttk.Button(front_buttons, text="✂️ Edit Crop", bootstyle=INFO, 
                  command=lambda: self.edit_crop("front")).pack(side=RIGHT, padx=2)
        
        # Back image controls
        back_frame = ttk.Frame(img_section)
        back_frame.pack(fill=X, pady=5)
        
        back_left = ttk.Frame(back_frame)
        back_left.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(back_left, text="Back Side (Optional):", font=("Helvetica", 10, "bold")).pack(anchor=W)
        self.back_name = ttk.Label(back_left, text="No file selected", foreground="#666")
        self.back_name.pack(anchor=W, pady=(2, 0))
        
        back_buttons = ttk.Frame(back_frame)
        back_buttons.pack(side=RIGHT)
        ttk.Button(back_buttons, text="📂 Browse", bootstyle=SECONDARY, command=self.load_back).pack(side=RIGHT, padx=2)
        ttk.Button(back_buttons, text="✂️ Edit Crop", bootstyle=INFO, 
                  command=lambda: self.edit_crop("back")).pack(side=RIGHT, padx=2)
        
        # Preview section
        preview_section = ttk.Labelframe(body, text="📋 Preview", padding=15)
        preview_section.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        preview_container = ttk.Frame(preview_section)
        preview_container.pack(fill=BOTH, expand=True)
        
        # Front preview
        front_preview_frame = ttk.Frame(preview_container)
        front_preview_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        ttk.Label(front_preview_frame, text="Front Preview", font=("Helvetica", 10, "bold")).pack()
        self.front_preview = ttk.Label(front_preview_frame, text="No image loaded", 
                                     background="#f8f9fa", relief="solid", borderwidth=1)
        self.front_preview.pack(fill=BOTH, expand=True, pady=5)
        
        # Back preview  
        back_preview_frame = ttk.Frame(preview_container)
        back_preview_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))
        ttk.Label(back_preview_frame, text="Back Preview", font=("Helvetica", 10, "bold")).pack()
        self.back_preview = ttk.Label(back_preview_frame, text="No image loaded", 
                                    background="#f8f9fa", relief="solid", borderwidth=1)
        self.back_preview.pack(fill=BOTH, expand=True, pady=5)

        # Settings section
        opts = ttk.Labelframe(body, text="⚙️ Print Settings", padding=15)
        opts.pack(fill=X, pady=(0, 10))

        # Checkboxes in a nice layout
        checkbox_frame = ttk.Frame(opts)
        checkbox_frame.pack(fill=X, pady=(0, 10))
        
        left_checks = ttk.Frame(checkbox_frame)
        left_checks.pack(side=LEFT, fill=X, expand=True)
        ttk.Checkbutton(left_checks, text="📐 Enforce standard card aspect (85.6 × 54 mm)", 
                       variable=self.enforce_ratio_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(left_checks, text="🔄 Rotate back images 180° (duplex printing)", 
                       variable=self.rotate_back_var).pack(anchor="w", pady=2)
        
        right_checks = ttk.Frame(checkbox_frame)
        right_checks.pack(side=RIGHT, fill=X, expand=True)
        ttk.Checkbutton(right_checks, text="✂️ Add cutting guide marks", 
                       variable=self.crop_marks_var).pack(anchor="w", pady=2)

        # Settings grid with better layout
        grid_frame = ttk.Frame(opts)
        grid_frame.pack(fill=X)
        
        # Left column - Card dimensions
        left_grid = ttk.Labelframe(grid_frame, text="Card Dimensions", padding=10)
        left_grid.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        
        ttk.Label(left_grid, text="Width (mm):").grid(row=0, column=0, sticky="e", padx=(0, 5), pady=3)
        ttk.Entry(left_grid, textvariable=self.card_w_mm, width=12).grid(row=0, column=1, sticky="w", pady=3)
        
        ttk.Label(left_grid, text="Height (mm):").grid(row=1, column=0, sticky="e", padx=(0, 5), pady=3)
        ttk.Entry(left_grid, textvariable=self.card_h_mm, width=12).grid(row=1, column=1, sticky="w", pady=3)
        
        # Right column - Layout settings
        right_grid = ttk.Labelframe(grid_frame, text="Layout Settings", padding=10)
        right_grid.pack(side=RIGHT, fill=X, expand=True, padx=(5, 0))
        
        ttk.Label(right_grid, text="Page margin (mm):").grid(row=0, column=0, sticky="e", padx=(0, 5), pady=3)
        ttk.Entry(right_grid, textvariable=self.margin_mm, width=12).grid(row=0, column=1, sticky="w", pady=3)
        
        ttk.Label(right_grid, text="Card spacing (mm):").grid(row=1, column=0, sticky="e", padx=(0, 5), pady=3)
        ttk.Entry(right_grid, textvariable=self.padding_mm, width=12).grid(row=1, column=1, sticky="w", pady=3)
        
        ttk.Label(right_grid, text="Crop mark offset (mm):").grid(row=2, column=0, sticky="e", padx=(0, 5), pady=3)
        ttk.Entry(right_grid, textvariable=self.crop_mark_offset_mm, width=12).grid(row=2, column=1, sticky="w", pady=3)

        # Action buttons with better styling
        ttk.Separator(self.root, orient=HORIZONTAL).pack(fill=X, pady=5)
        
        actions = ttk.Frame(self.root, padding=15)
        actions.pack(fill=X)
        
        # Progress bar (hidden initially)
        self.progress_bar = ttk.Progressbar(actions, variable=self.progress_var, mode='determinate')
        
        button_frame = ttk.Frame(actions)
        button_frame.pack(fill=X)
        
        ttk.Button(button_frame, text="🚀 Generate PDF Files", bootstyle=SUCCESS, 
                  command=self.generate_pdfs, width=20).pack(side=LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="📁 Open Output Folder", bootstyle=INFO, 
                  command=self.open_output_folder, width=18).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Exit", bootstyle=SECONDARY, 
                  command=self.root.quit, width=10).pack(side=RIGHT)
        
        self.output_folder = None
        
    def open_output_folder(self):
        if self.output_folder and os.path.exists(self.output_folder):
            if os.name == 'nt':  # Windows
                os.startfile(self.output_folder)
            elif os.name == 'posix':  # macOS and Linux
                os.system(f'open "{self.output_folder}"' if 'darwin' in os.uname().sysname.lower() else f'xdg-open "{self.output_folder}"')
        else:
            messagebox.showinfo("No Output Folder", "Generate PDFs first to create an output folder.")

    # -------------------------
    # Loaders + previews
    # -------------------------
    def load_front(self):
        path = filedialog.askopenfilename(
            title="Select Front Image",
            filetypes=[
                ("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tiff;*.tif"),
                ("JPEG Files", "*.jpg;*.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            return
            
        self.status_text.set("Processing front image...")
        self.root.update()
        
        try:
            self.front_path = path
            filename = os.path.basename(path)
            if len(filename) > 40:
                filename = filename[:37] + "..."
            self.front_name.configure(text=filename, foreground="#2d8a2f")
            
            img = crop_card_with_optional_manual(
                path, manual_box=None, 
                enforce_ratio=self.enforce_ratio_var.get(),
                ratio_w=self.card_w_mm.get(), 
                ratio_h=self.card_h_mm.get()
            )
            self.front_preview_img = img
            self.front_manual_box = None
            self._show_preview("front")
            self.status_text.set("Front image loaded successfully")
            
        except Exception as e:
            self.status_text.set(f"Error loading front image: {str(e)[:50]}...")
            messagebox.showerror("Front Image Error", f"Failed to load front image:\n\n{str(e)}")

    def load_back(self):
        path = filedialog.askopenfilename(
            title="Select Back Image (Optional)",
            filetypes=[
                ("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tiff;*.tif"),
                ("JPEG Files", "*.jpg;*.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            return
            
        self.status_text.set("Processing back image...")
        self.root.update()
        
        try:
            self.back_path = path
            filename = os.path.basename(path)
            if len(filename) > 40:
                filename = filename[:37] + "..."
            self.back_name.configure(text=filename, foreground="#2d8a2f")
            
            img = crop_card_with_optional_manual(
                path, manual_box=None, 
                enforce_ratio=self.enforce_ratio_var.get(),
                ratio_w=self.card_w_mm.get(), 
                ratio_h=self.card_h_mm.get()
            )
            self.back_preview_img = img
            self.back_manual_box = None
            self._show_preview("back")
            self.status_text.set("Back image loaded successfully")
            
        except Exception as e:
            self.status_text.set(f"Error loading back image: {str(e)[:50]}...")
            messagebox.showerror("Back Image Error", f"Failed to load back image:\n\n{str(e)}")

    def _show_preview(self, which):
        lbl = self.front_preview if which == "front" else self.back_preview
        pil = self.front_preview_img if which == "front" else self.back_preview_img
        
        if pil is None:
            lbl.configure(image="", text=f"No {which} image loaded", background="#f8f9fa")
            return
            
        # Create a better preview with consistent sizing
        show = pil.copy()
        preview_size = (280, 180) if which == "front" else (280, 180)
        
        # Calculate aspect ratio preserving resize
        w, h = show.size
        max_w, max_h = preview_size
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        show = show.resize((new_w, new_h), Image.LANCZOS)
        
        # Add a border for better visual separation
        bordered = Image.new("RGB", (max_w, max_h), color="white")
        paste_x = (max_w - new_w) // 2
        paste_y = (max_h - new_h) // 2
        bordered.paste(show, (paste_x, paste_y))
        
        tkimg = ImageTk.PhotoImage(bordered)
        if which == "front":
            self.front_preview_tk = tkimg
        else:
            self.back_preview_tk = tkimg
        lbl.configure(image=tkimg, text="", background="white")

    # -------------------------
    # Manual crop editing on ORIGINAL image
    # -------------------------
    def edit_crop(self, which):
        src_path = self.front_path if which == "front" else self.back_path
        if not src_path:
            messagebox.showwarning("No file", "Select the image first.")
            return
        # open original image as PIL
        orig = Image.open(src_path).convert("RGB")
        editor = RoiEditor(self.root, orig, title=f"Draw crop on ORIGINAL — {which.upper()}")
        self.root.wait_window(editor)
        if editor.crop_box:
            # save manual box in original coords
            if which == "front":
                self.front_manual_box = editor.crop_box
            else:
                self.back_manual_box = editor.crop_box
            # Apply crop -> detection inside manual box w/ perspective when possible
            try:
                pil = crop_card_with_optional_manual(src_path,
                                                    manual_box=editor.crop_box,
                                                    enforce_ratio=self.enforce_ratio_var.get(),
                                                    ratio_w=self.card_w_mm.get(),
                                                    ratio_h=self.card_h_mm.get())
                if which == "front":
                    self.front_preview_img = pil
                else:
                    self.back_preview_img = pil
                self._show_preview(which)
            except Exception as e:
                messagebox.showerror("Crop apply error", str(e))

    # -------------------------
    # Generate PDFs
    # -------------------------
    def generate_pdfs(self):
        if self.front_preview_img is None:
            messagebox.showerror("Missing Image", "Please load a front image before generating PDFs.")
            return

        save_dir = filedialog.askdirectory(title="Select Output Folder")
        if not save_dir:
            return
            
        self.output_folder = save_dir
        self.status_text.set("Generating PDF files...")
        self.progress_bar.pack(fill=X, pady=(10, 0))
        self.progress_var.set(10)
        self.root.update()

        # Ensure front/back same pixel size
        if self.back_preview_img is not None:
            fw, fh = self.front_preview_img.size
            bw, bh = self.back_preview_img.size
            if (bw, bh) != (fw, fh):
                self.back_preview_img = self.back_preview_img.resize((fw, fh), Image.LANCZOS)
                self._show_preview("back")

        try:
            # Create temporary files directory
            self.progress_var.set(20)
            self.status_text.set("Preparing images...")
            self.root.update()
            
            # Create temp directory in the project folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            temp_dir = os.path.join(script_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            tmp_front = os.path.join(temp_dir, "front_temp.png")
            self.front_preview_img.save(tmp_front)
            tmp_back = None
            if self.back_preview_img is not None:
                back_to_save = self.back_preview_img.rotate(180) if self.rotate_back_var.get() else self.back_preview_img
                tmp_back = os.path.join(temp_dir, "back_temp.png")
                back_to_save.save(tmp_back)

            # card size in points from mm
            self.progress_var.set(40)
            self.status_text.set("Calculating layout...")
            self.root.update()
            
            card_w_pt = mm_to_pt(float(self.card_w_mm.get()))
            card_h_pt = mm_to_pt(float(self.card_h_mm.get()))
            spacing_mm = float(self.padding_mm.get())
            margin_mm = float(self.margin_mm.get())

            # choose orientation that yields more cards
            rows_p, cols_p = layout_counts(A4[0], A4[1], card_w_pt, card_h_pt, mm_to_pt(margin_mm), mm_to_pt(spacing_mm))
            rows_l, cols_l = layout_counts(A4[1], A4[0], card_w_pt, card_h_pt, mm_to_pt(margin_mm), mm_to_pt(spacing_mm))
            count_p = rows_p * cols_p
            count_l = rows_l * cols_l
            orientation = "landscape" if count_l > count_p else "portrait"
            cards_per_page = count_l if orientation == "landscape" else count_p

            # Generate front PDF
            self.progress_var.set(60)
            self.status_text.set("Generating front PDF...")
            self.root.update()
            
            front_pdf = os.path.join(save_dir, "id_card_fronts.pdf")
            draw_tiled_pdf(front_pdf, tmp_front, card_w_pt, card_h_pt, margin_mm=margin_mm, spacing_mm=spacing_mm,
                           orientation=orientation, crop_marks=self.crop_marks_var.get(), 
                           crop_mark_offset_mm=self.crop_mark_offset_mm.get())

            if tmp_back:
                self.progress_var.set(80)
                self.status_text.set("Generating back PDF...")
                self.root.update()
                
                back_pdf = os.path.join(save_dir, "id_card_backs.pdf")
                draw_tiled_pdf(back_pdf, tmp_back, card_w_pt, card_h_pt, margin_mm=margin_mm, spacing_mm=spacing_mm,
                               orientation=orientation, crop_marks=self.crop_marks_var.get(),
                               crop_mark_offset_mm=self.crop_mark_offset_mm.get())
                
                self.progress_var.set(100)
                self.status_text.set("PDFs generated successfully!")
                
                messagebox.showinfo("✅ Generation Complete!", 
                    f"ID Card PDFs created successfully!\n\n"
                    f"📄 Front: id_card_fronts.pdf\n"
                    f"📄 Back: id_card_backs.pdf\n\n"
                    f"📐 Layout: {orientation.title()} ({cards_per_page} cards per page)\n"
                    f"📏 Card size: {self.card_w_mm.get()}×{self.card_h_mm.get()} mm\n"
                    f"📁 Location: {save_dir}")
            else:
                self.progress_var.set(100)
                self.status_text.set("PDF generated successfully!")
                
                messagebox.showinfo("✅ Generation Complete!", 
                    f"ID Card PDF created successfully!\n\n"
                    f"📄 Front: id_card_fronts.pdf\n\n"
                    f"📐 Layout: {orientation.title()} ({cards_per_page} cards per page)\n"
                    f"📏 Card size: {self.card_w_mm.get()}×{self.card_h_mm.get()} mm\n"
                    f"📁 Location: {save_dir}")

            # cleanup temp images
            try:
                if os.path.exists(tmp_front):
                    os.remove(tmp_front)
                if tmp_back and os.path.exists(tmp_back):
                    os.remove(tmp_back)
            except Exception:
                pass
                
        except Exception as e:
            self.status_text.set(f"Error generating PDFs")
            messagebox.showerror("Generation Error", f"Failed to generate PDFs:\n\n{str(e)}")
        finally:
            # Hide progress bar after completion
            self.root.after(3000, lambda: self.progress_bar.pack_forget())


def main():
    app = ttk.Window(themename="flatly")
    IDCardApp(app)
    app.mainloop()


if __name__ == "__main__":
    main()
