// read_csv.cpp

#include "read_csv.h" // Must be first for precompiled headers, or just good practice
#include <fstream>
#include <sstream>
#include <iostream> // For std::cerr, std::cout

// Helper function to split a string by a delimiter (static linkage is fine within this .cpp)
static std::vector<std::string> splitString(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

// --- Original function implementation (can remain as is) ---
std::vector<std::vector<std::string>> readEntireCSV(const std::string &filename, std::vector<std::string> &header)
{
    std::vector<std::vector<std::string>> data;
    header.clear(); // Ensure header is empty if function fails early

    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data; // Return empty data
    }

    std::string line;

    // Read header line
    if (std::getline(file, line))
    {
        if (line.empty() && file.eof())
        { // Handle completely empty file
            std::cerr << "Error: File " << filename << " is empty." << std::endl;
            file.close();
            return data;
        }
        header = splitString(line, ',');
    }
    else
    {
        std::cerr << "Error: Could not read header from file " << filename << std::endl;
        file.close();
        return data; // Return empty data
    }

    if (header.empty())
    {
        std::cerr << "Warning: Header line in " << filename << " is empty or could not be parsed." << std::endl;
    }

    // Read data rows
    while (std::getline(file, line))
    {
        if (line.empty() && file.eof())
        { // Skip trailing empty lines
            continue;
        }
        if (line.empty())
        { // Skip empty lines within the data
            continue;
        }
        data.push_back(splitString(line, ','));
    }

    file.close();
    return data;
}

// --- NEW/MODIFIED FUNCTION IMPLEMENTATIONS ---

// General CSV reader
CSVData readCSV(const std::string &filename, bool firstLineIsHeader)
{
    CSVData result;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return result; // Return empty CSVData
    }

    std::string line;

    // Handle header if specified
    if (firstLineIsHeader)
    {
        if (std::getline(file, line))
        {
            if (line.empty() && file.eof())
            { // Handle completely empty file
                std::cerr << "Error: File " << filename << " is empty." << std::endl;
                file.close();
                return result;
            }
            result.header = splitString(line, ',');
            if (result.header.empty())
            {
                std::cerr << "Warning: Header line in " << filename << " is empty or could not be parsed." << std::endl;
            }
        }
        else
        {
            std::cerr << "Error: Could not read header from file " << filename << " (or file is empty)." << std::endl;
            file.close();
            return result; // Return empty CSVData
        }
    }

    // Read data rows
    // If firstLineIsHeader was false, the first getline here will read the first line of the file.
    while (std::getline(file, line))
    {
        if (line.empty())
        { // Skip empty lines (trailing or internal)
            if (file.eof())
                break; // Avoid issues if last line is empty
            // std::cerr << "Warning: Skipping empty line in " << filename << std::endl; // Optional
            continue;
        }
        result.rows.push_back(splitString(line, ','));
    }

    file.close();
    return result;
}

// Convenience function: Assumes first line is header
CSVData readCSVWithHeader(const std::string &filename)
{
    return readCSV(filename, true);
}

// Convenience function: Assumes no header, all lines are data
CSVData readCSVDataOnly(const std::string &filename)
{
    return readCSV(filename, false);
}

// Modified printCSVPreview
void printCSVPreview(const std::string &filename, bool hasHeader, int numDataRowsToPrint)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for preview." << std::endl;
        return;
    }

    std::cout << "CSV Preview for: " << filename << std::endl;
    std::string line;

    if (hasHeader)
    {
        if (std::getline(file, line))
        {
            std::cout << "Header: " << line << std::endl;
        }
        else
        {
            std::cout << "File '" << filename << "' is empty or header could not be read." << std::endl;
            file.close();
            return;
        }
    }
    else
    {
        std::cout << "(No header row assumed/printed)" << std::endl;
    }

    std::cout << "First " << numDataRowsToPrint << " data rows:" << std::endl;
    int rowsPrinted = 0;
    for (int i = 0; i < numDataRowsToPrint && std::getline(file, line); /* ++i handled below */)
    {
        if (line.empty())
        {
            if (file.eof())
                break; // Skip trailing empty line
            // std::cout << "(Skipping empty line)" << std::endl; // Optional
            continue; // Don't count this as a printed data row, don't increment i
        }
        std::cout << "Row " << rowsPrinted + 1 << ":  " << line << std::endl;
        rowsPrinted++;
        i++; // Increment i only when a non-empty row is processed
    }

    if (rowsPrinted == 0 && numDataRowsToPrint > 0)
    {
        std::cout << "(No data rows found";
        if (hasHeader)
            std::cout << " after header";
        std::cout << " or all were empty)" << std::endl;
    }
    else if (rowsPrinted < numDataRowsToPrint && file.eof() && rowsPrinted > 0) // Check rowsPrinted > 0 to avoid double message
    {
        std::cout << "(Reached end of file after " << rowsPrinted << " data rows)" << std::endl;
    }
    std::cout << "--- End of Preview ---" << std::endl;

    file.close();
}