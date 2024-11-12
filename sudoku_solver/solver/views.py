import cv2
import numpy as np
import easyocr
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

        self.solve_sudoku(extracted_numbers)
        
        # Respond with the extracted numbers
        return JsonResponse({'extracted_numbers': extracted_numbers}, status=status.HTTP_200_OK)
    
    def solve_sudoku(self, board):
        # Find the next empty cell in the puzzle
        empty_cell = self.find_empty_location(board)
        if not empty_cell:
            # No empty cell means the puzzle is solved
            return True
        row, col = empty_cell

        # Try numbers 1 through 9 in the empty cell
        for num in range(1, 10):
            if self.is_safe(board, row, col, num):
                board[row][col] = num  # Place number in the cell

                # Recursively try to solve the puzzle
                if self.solve_sudoku(board):
                    return True

                # If placing num doesn't lead to a solution, reset the cell
                board[row][col] = 0

        return False  # Trigger backtracking if no solution is found


    def find_empty_location(self, board):
        # Find the next cell in the puzzle that is empty (represented by 0)
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    return (row, col)
        return None  # Return None if there are no empty cells


    def is_safe(self, board, row, col, num):
        # Check if num is not in the current row
        for i in range(9):
            if board[row][i] == num:
                return False

        # Check if num is not in the current column
        for i in range(9):
            if board[i][col] == num:
                return False

        # Check if num is not in the current 3x3 subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False

        return True  # The number can be safely placed
    
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
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        
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
                
                # Use EasyOCR to extract the number from the cell
                result = reader.readtext(thresh_roi)
                
                # Filter the detected text to extract only numeric values
                detected_text = ''
                for detection in result:
                    detected_text = detection[1]  # Getting the text part of the detection
                
                try:
                    # If a number is found, append it to the list
                    num = int(detected_text.strip())
                    row_numbers.append(num)
                except ValueError:
                    # If no number is found, append 0 (for empty cells)
                    row_numbers.append(0)
            
            # Add the row of numbers to the full grid
            numbers.append(row_numbers)
        
        return numbers
