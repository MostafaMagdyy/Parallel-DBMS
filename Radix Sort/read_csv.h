// read_csv.h

#ifndef READ_CSV_H
#define READ_CSV_H

#include <string>
#include <vector>
#include <iostream> // For potential error messages from within header if inlined

// (Keep your CSVData struct definition here or include its header)
struct CSVData
{
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
};

// Original function - still useful if you prefer the out-parameter style
std::vector<std::vector<std::string>> readEntireCSV(const std::string &filename, std::vector<std::string> &header);

// --- NEW/MODIFIED FUNCTIONS ---

// Function to read CSV, assumes first line is header.
// Returns header and data in a CSVData struct.
CSVData readCSVWithHeader(const std::string &filename);

// Function to read CSV, assumes NO header row. All lines are data.
// Returns only data rows; the 'header' field in CSVData will be empty.
CSVData readCSVDataOnly(const std::string &filename);

// A more general function.
// If 'firstLineIsHeader' is true, the first line is parsed as header.
// Otherwise, all lines are treated as data.
CSVData readCSV(const std::string &filename, bool firstLineIsHeader);

// Function to print the header and the first few data rows of a CSV file.
// 'numDataRowsToPrint' is the number of data rows to print AFTER the header.
// 'hasHeader' indicates if the file is expected to have a header row.
void printCSVPreview(const std::string &filename, bool hasHeader = true, int numDataRowsToPrint = 5);

#endif // READ_CSV_H