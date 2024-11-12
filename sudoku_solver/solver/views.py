import cv2
import numpy as np
import pytesseract
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from rest_framework import status

from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')


class SudokuSolverView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # Get the uploaded image
        file = request.FILES['image']
        image = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Detect the Sudoku grid
        sudoku_grid = self.detect_sudoku_grid(img)
        
        # Extract digits from the grid using OCR
        extracted_numbers = self.extract_numbers_from_grid(img, sudoku_grid)
        
        # Respond with the extracted numbers
        return JsonResponse({'extracted_numbers': extracted_numbers}, status=status.HTTP_200_OK)

    def detect_sudoku_grid(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours to locate the Sudoku grid
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Placeholder: Assuming the Sudoku grid is the largest contour
        grid_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the grid's corners (4 points)
        epsilon = 0.02 * cv2.arcLength(grid_contour, True)
        approx = cv2.approxPolyDP(grid_contour, epsilon, True)
        
        # Get the four corners of the grid
        if len(approx) == 4:
            pts = np.array(approx, dtype="float32")
            return pts
        else:
            return None

    def extract_numbers_from_grid(self, img, grid_pts):
        # Define the dimensions of each cell
        cell_width = img.shape[1] // 9
        cell_height = img.shape[0] // 9
        
        numbers = []
        
        # Process each cell in the Sudoku grid
        for row in range(9):
            row_numbers = []
            for col in range(9):
                # Define the region of interest (ROI) for each cell
                x = col * cell_width
                y = row * cell_height
                roi = img[y:y + cell_height, x:x + cell_width]
                
                # Convert ROI to grayscale and apply thresholding
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV)
                
                # Use pytesseract to extract the number from the cell
                text = pytesseract.image_to_string(thresh_roi, config='--psm 6 --oem 3', lang='eng')
                try:
                    # If a number is found, append it to the list
                    num = int(text.strip())
                    row_numbers.append(num)
                except ValueError:
                    # If no number is found, append 0 (for empty cells)
                    row_numbers.append(0)
            
            # Add the row of numbers to the full grid
            numbers.append(row_numbers)
        
        return numbers