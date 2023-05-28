import os
import tkinter as tk
import customtkinter as ctk
from customtkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
import heapq
from collections import defaultdict
from zigzag import *
import gzip
import ast

class ImageCompressorGUI:

    gz=False
    txt=False
    color_blue="#1F6AA5"
    color_green="#22B14C"
    color_purple="#A349A4"
    color_orange="#FF7F27"
    color_brown="#F08784"
    color_frame="#2B2B2B"
    color_root="#242424"
    #Block Size
    block_size = 8
    # For High-quality JPEG compression (Lower values)
    # QUANTIZATION_MAT = np.array([[3, 2, 2, 3, 5, 8, 10, 12],
    #                         [2, 2, 3, 4, 5, 12, 12, 11],
    #                         [3, 3, 3, 5, 8, 11, 14, 11],
    #                         [3, 3, 4, 6, 10, 17, 16, 12],
    #                         [4, 4, 7, 11, 14, 22, 21, 15],
    #                         [5, 7, 11, 13, 16, 12, 23, 18],
    #                         [10, 13, 16, 17, 21, 24, 24, 21],
    #                         [14, 18, 19, 20, 22, 20, 20, 20]])
    # For Balanced-quality JPEG compression
    QUANTIZATION_MAT = np.array([[8, 6, 5, 8, 12, 20, 26, 31],
                           [6, 6, 7, 10, 13, 29, 30, 28],
                           [7, 6, 8, 12, 20, 29, 35, 28],
                           [7, 9, 11, 14, 26, 44, 40, 31],
                           [9, 11, 19, 28, 34, 55, 52, 39],
                           [12, 17, 27, 32, 41, 52, 57, 46],
                           [24, 32, 39, 44, 52, 61, 60, 51],
                           [36, 46, 48, 49, 56, 50, 51, 50]])
    #Default (poor quality for lower gray levels)
    # QUANTIZATION_MAT = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61],
    #                         [ 12,  12,  14,  19,  26,  58,  60,  55],
    #                         [ 14,  13,  16,  24,  40,  57,  69,  56],
    #                         [ 14,  17,  22,  29,  51,  87,  80,  62],
    #                         [ 18,  22,  37,  56,  68, 109, 103,  77],
    #                         [ 24,  35,  55,  64,  81, 104, 113,  92],
    #                         [ 49,  64,  78,  87, 103, 121, 120, 101],
    #                         [ 72,  92,  95,  98, 112, 100, 103,  99]])

    
    def __init__(self, master):
        self.master = master
        self.master.title("JPEG Image Compressor")
        self.master.geometry("850x735")

        self.image_path = None
        self.quality = None

        img_frame = ctk.CTkFrame(self.master)
        img_frame.pack(pady=10,padx=10,fill='both',expand=True)

        self.original_image_label = ctk.CTkLabel(img_frame, text="Original Image")
        self.original_image_label.pack(side=ctk.LEFT, padx=10)
        self.original_image = tk.Label(img_frame, background="#2B2B2B")
        self.original_image.pack(side=tk.LEFT)

        self.compressed_image_label = ctk.CTkLabel(
            img_frame, text="Compressed Image")
        self.compressed_image_label.pack(side=ctk.RIGHT, padx=10)
        self.compressed_image = tk.Label(img_frame, background="#2B2B2B")
        self.compressed_image.pack(side=tk.RIGHT)
        ####Select Image####
        self.bar= ctk.CTkProgressBar(self.master, width=250, mode="indeterminate")
        self.bar.pack(pady=1)
        self.bar.configure(fg_color="#242424")
        self.bar.configure(progress_color="#242424")
        select_image_button = ctk.CTkButton(
            self.master, text="Select Image", command=self._select_image)
        select_image_button.pack(side=ctk.TOP, pady=10)
        self.size_lable_orignal = ctk.CTkLabel(master=self.master, text="", text_color="silver",)
        self.size_lable_orignal.pack(side=ctk.TOP)
        self.size_lable_compressed = ctk.CTkLabel(master=self.master, text="", text_color="silver",)
        self.size_lable_compressed.pack(side=ctk.TOP)
        ####Comprassor####
        btn_frame = ctk.CTkFrame(self.master)
        btn_frame.pack(pady=3,padx=10,fill='both')

        lable = ctk.CTkLabel(master=btn_frame, text="Comprassor", text_color="silver",)
        lable.pack(padx=10, pady=10)
        quality_frame = ctk.CTkFrame(btn_frame)
        quality_frame.pack(side=tk.LEFT, pady=10,padx=100)
        quality_label = ctk.CTkLabel(quality_frame, text="Quality (1-100)  ",bg_color="#2B2B2B")
        quality_label.pack(side=ctk.LEFT)
        self.quality_entry = ctk.CTkEntry(quality_frame)
        self.quality_entry.pack(side=ctk.LEFT)
        # Compress Image Button
        self.compress_image_button = ctk.CTkButton(
            btn_frame, text="Compress Image", command=self._compress_image)
        self.compress_image_button.pack(side=ctk.RIGHT, pady=10,padx=100)
        self.compress_image_button.configure(state="disabled")
        ####Decomprassor####
        lable = ctk.CTkLabel(master=self.master, text="Decomprassor", text_color="silver",)
        lable.pack(padx=10, pady=10)
        select_txt_button = ctk.CTkButton(
            self.master, text="Select Text File", command=self._select_txt)
        select_txt_button.pack(pady=5,padx=10)
        
        self.txt_path_lable = ctk.CTkLabel(master=self.master, text="", text_color="silver",)
        self.txt_path_lable.pack()
        self.txt_path_lable.configure(text="No .txt File Selected")

        select_gz_button = ctk.CTkButton(
            self.master, text="Select GZ File", command=self._select_gz)
        select_gz_button.pack(pady=5,padx=10)

        self.gz_path_lable = ctk.CTkLabel(master=self.master, text="", text_color="silver",)
        self.gz_path_lable.pack()
        self.gz_path_lable.configure(text="No .gz File Selected")

        self.decompress_image_button = ctk.CTkButton(
            self.master, text="Decompress Image", command=self._decompress_image)
        self.decompress_image_button.pack(side=ctk.RIGHT, pady=10,padx=10)
        self.decompress_image_button.configure(state="disabled")

    def _select_image(self):
        self.image_path = filedialog.askopenfilename(initialdir=os.getcwd(
        ), title="Select Image", filetypes=(("JPEG Files", "*.jpeg"), ("JPEG Files", "*.jpg")))

        if self.image_path:
            image = Image.open(self.image_path)

            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.original_image.config(image=photo)
            self.original_image.image = photo
            self.compress_image_button.configure(state="normal")
            self.size_lable_orignal.configure(text=f"Size: \t{os.path.getsize(self.image_path)/1024 :.3f} KB")
        else:
            self.original_image.config(image="")
            self.compress_image_button.configure(state="disabled")
            self.size_lable_orignal.configure(text="")
            self.size_lable_compressed.configure(text="")

    def _select_txt(self):
        # Ask the user to select a text file
        self.txt_file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        # Print the selected file path to the console
        #print("Selected file:", file_path)
        if(self.gz & bool(self.txt_file_path)):
            self.decompress_image_button.configure(state="normal")
            self.txt_path_lable.configure(text=self.txt_file_path)
        elif(self.txt_file_path):
            self.txt_path_lable.configure(text=self.txt_file_path)
            self.txt=True
        else:
            self.decompress_image_button.configure(state="disabled")
            self.txt_path_lable.configure(text="No File Selected")
            self.txt=False

    def _select_gz(self):
        # Ask the user to select a text file
        self.gz_file_path = filedialog.askopenfilename(filetypes=[("GZ Files", "*.gz")])
        # Print the selected file path to the console
        #print("Selected file:", file_path)
        if(bool(self.gz_file_path) & self.txt):
            self.decompress_image_button.configure(state="normal")
            self.gz_path_lable.configure(text=self.gz_file_path)
        elif(self.gz_file_path):
            self.gz_path_lable.configure(text=self.gz_file_path)
            self.gz=True
        else:
            self.decompress_image_button.configure(state="disabled")
            self.gz_path_lable.configure(text="No File Selected") 
            self.gz=False       

    def get_run_length_coding(self,image):
        #self.bar.configure(fg_color=self.color_frame)
        self.bar.configure(progress_color=self.color_green)
        i = 0
        skip = 0
        stream = []    
        bitstream = ""
        image = image.astype(int)
        shapeImg=image.shape[0]
        while i < shapeImg:
            if image[i] != 0:            
                stream.append((image[i],skip))
                bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
                skip = 0
            else:
                skip = skip + 1
            i = i + 1
            root.update()
            #self.bar.set(float(i/shapeImg))
            #root.update_idletasks()
        #self.bar.configure(fg_color=self.color_root)
        #self.bar.configure(progress_color=self.color_root)
        #root.update_idletasks()
        return bitstream
    
    def encode(self, s):
        self.bar.configure(progress_color=self.color_orange)
        freq = defaultdict(int)
        for c in s:
            freq[c] += 1
        
        h = [[f, [c, ""]] for c, f in freq.items()]
        heapq.heapify(h)
        
        while len(h) > 1:
            lo = heapq.heappop(h)
            hi = heapq.heappop(h)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(h, [lo[0] + hi[0]] + lo[1:] + hi[1:])
            root.update()
        
        return dict(h[0][1:])

    def decode(self, encoded, s):
        self.bar.configure(progress_color=self.color_brown)
        reverse = {v:k for k, v in encoded.items()}
        output = ""
        i = 0
        while i < len(s):
            for k in reverse:
                root.update()
                if s[i:].startswith(k):
                    output += reverse[k]
                    i += len(k)
        return output

    def read_txt_file(self, filename):
        try:
            with open(filename, 'r') as file:
                data = file.read()
            dict_obj = ast.literal_eval(data)
        except SyntaxError:
            messagebox.showerror("Error", "Wrong Text file Selected")
            return
        return dict_obj

    def _compress_image(self):
        
        self.bar.configure(fg_color=self.color_frame)
        self.bar.configure(progress_color=self.color_blue)
        self.bar.start()
        root.update_idletasks()
        quality= self.quality_entry.get()
        try:
            quality = int(quality)
            if quality < 1 or quality > 100:
                messagebox.showerror(
                    "Error", "Quality value must be between 1-100")
                return
        except ValueError:
            messagebox.showerror("Error", "Quality value must be a number")
            return
        
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)     # grayscale style
        [h , w] = img.shape
        height = h
        width = w
        h = np.float32(h)
        w = np.float32(w)
        nbh = math.ceil(h/self.block_size)
        nbh = np.int32(nbh)
        nbw = math.ceil(w/self.block_size)
        nbw = np.int32(nbw)
        # Pad the image, to dividable to block size
        H =  self.block_size * nbh
        W =  self.block_size * nbw
        padded_img = np.zeros((H,W))
        padded_img[0:height,0:width] = img[0:height,0:width]
        image = Image.fromarray(np.uint8(padded_img))
        #display
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.original_image.config(image=photo)
        self.original_image.image = photo
        cv2.imwrite('UnCompressed.bmp', np.uint8(padded_img))
        self.size_lable_orignal.configure(text=f"Orignal Image Size :   \t{os.path.getsize('UnCompressed.bmp')/(1024):.3f} KB")
        # start encoding:
        for i in range(nbh):
                row_ind_1 = i*self.block_size                
                row_ind_2 = row_ind_1+self.block_size
                for j in range(nbw):
                    col_ind_1 = j*self.block_size                       
                    col_ind_2 = col_ind_1+self.block_size
                    block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]                     
                    DCT = cv2.dct(block)            
                    DCT_normalized = np.divide(DCT,self.QUANTIZATION_MAT * (quality/100)).astype(int)            
                    reordered = zigzag(DCT_normalized)
                    reshaped= np.reshape(reordered, (self.block_size, self.block_size)) 
                    padded_img[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshaped 
                root.update()
                #if(i%10==0):
        #root.update_idletasks()
        #cv2.imshow('encoded image', np.uint8(padded_img))
        self.compressed_image_label.configure(text="Encoded Image")
        image = Image.fromarray(np.uint8(padded_img))
        #display
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.compressed_image.config(image=photo)
        self.compressed_image.image = photo
        arranged = padded_img.flatten()
        #RLE encoded data is written to a text file
        bitstream = self.get_run_length_coding(arranged)
        bitstream = str(padded_img.shape[0]) + " " + str(padded_img.shape[1]) + " " + bitstream + ";"
        encoded =  self.encode(bitstream)
        huffman_codes = ''.join([encoded[c] for c in bitstream])
        self.size_lable_compressed.configure(text=f"Compressed Image Size :\t{(int(len(huffman_codes)/8))/(1024):.3f} KB")
        # Written to dictionary.txt (compressed)
        file2 = open("dictionary.txt","w")
        file2.write(str(encoded))
        file2.close()
        # Written to codes.txt (compressed)
        with gzip.open('huffman_codes.gz', 'wt') as f:
            f.write(huffman_codes)
        self.bar.configure(fg_color=self.color_root)
        self.bar.configure(progress_color=self.color_root)
        self.bar.stop()

    def _decompress_image(self):

        self.bar.configure(fg_color=self.color_frame)
        self.bar.configure(progress_color=self.color_purple)
        
        #read files
        try:
            encoded = self.read_txt_file(self.txt_file_path)
        except SyntaxError:
            messagebox.showerror("Error", "Wrong Text file Selected")
            return
        try:
            with gzip.open('huffman_codes.gz', 'rt') as f:
                huffman_codes = f.read()
        except SyntaxError:
            messagebox.showerror("Error", "Wrong GZ file Selected")
            return
        try:
            decoded = self.decode(encoded, huffman_codes)
        except AttributeError:
            messagebox.showerror("Error", "Wrong files Selected")
            return
        self.bar.configure(progress_color=self.color_purple)
        root.update_idletasks()
        image= str(decoded)
        #tokanize
        details = image.split()
        h = int(''.join(filter(str.isdigit, details[0])))
        w = int(''.join(filter(str.isdigit, details[1])))
        array = np.zeros(h*w).astype(int)
        k = 0
        i = 2
        x = 0
        j = 0
        while k < array.shape[0]:
            #last char check
            if(details[i] == ';'):
                break
            if "-" not in details[i]:
                array[k] = int(''.join(filter(str.isdigit, details[i])))        
            else:
                array[k] = -1*int(''.join(filter(str.isdigit, details[i])))        
            if(i+3 < len(details)):
                j = int(''.join(filter(str.isdigit, details[i+3])))
            if j == 0:
                k = k + 1
            else:                
                k = k + j + 1        
            i = i + 2
            root.update()
        array = np.reshape(array,(h,w))
        # loop for constructing intensity matrix form frequency matrix (IDCT)
        i = 0
        j = 0
        k = 0
        padded_img = np.zeros((h,w))
        while i < h:
            j = 0
            while j < w:        
                temp_stream = array[i:i+8,j:j+8]                
                block = inverse_zigzag(temp_stream.flatten(), int(self.block_size),int(self.block_size))            
                de_quantized = np.multiply(block,self.QUANTIZATION_MAT)                
                padded_img[i:i+8,j:j+8] = cv2.idct(de_quantized)        
                j = j + 8        
            i = i + 8
            root.update()
        #root.update_idletasks()
        padded_img[padded_img > 255] = 255
        padded_img[padded_img < 0] = 0
        self.compressed_image_label.configure(text="Deompressed Image")
        image = Image.fromarray(np.uint8(padded_img))
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.compressed_image.config(image=photo)
        self.compressed_image.image = photo
        cv2.imwrite("DeCompressed_image.bmp",np.uint8(padded_img))
        self.bar.configure(fg_color=self.color_root)
        self.bar.configure(progress_color=self.color_root)
        self.bar.stop()

    def run(self):
        self.master.mainloop()

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = ImageCompressorGUI(root)
    app.run()
