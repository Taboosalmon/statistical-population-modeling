#include <iostream>
#include <math.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <unordered_map>

using std::string;
using std::cout;
using std::cin;
using std::array;
using std::vector;

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct country_data
{
	string country_name;
	string country_id;
	int gdp_capita;
	int life_expectancy;
	int healthcare;
	int temperature;
	int area;
	int landlocked;
	int population;
};

std::vector<country_data> out_p;
std::unordered_map<string, int> country_index_by_id;
int k = 1;




void load_data()
{
    std::ifstream file("../read_in.csv");
    
    if (!file.is_open()) {
        std::cerr << "Error opening file.\n";
        exit(1);
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<string> row;
        bool no_yeah = true;

        while (std::getline(ss, cell, ','))
        {
            if (cell == "..")
            {
                no_yeah = false;
            }
            row.push_back(cell);
        }

        if (no_yeah == true)
        {
            //country name,country id,gdp per capita,life expectancy,healthcare,population,landlocked,temperature, area
            country_data count_p;
            count_p.country_name = row[0];
            count_p.country_id = row[1];
            count_p.gdp_capita = std::stof(row[2]);
            count_p.life_expectancy = std::stof(row[3]);
            count_p.healthcare = std::stof(row[4]);
            count_p.population = std::stof(row[5]);
            count_p.landlocked = std::stof(row[6]);
            count_p.temperature = std::stoi(row[7]);
            count_p.area = std::stoi(row[8]);
            country_index_by_id[count_p.country_id] = out_p.size();
            out_p.push_back(count_p);
        }

    }

    file.close();
}

double eval_f(country_data& p2, VectorXd& x)
{
    double sum = x(k*6);
    for (int q = 1; q <= k; q++)
    {
        sum += x[6*q - 6] * pow(log10(p2.gdp_capita), q);
        sum += x[6*q - 5] * pow(p2.life_expectancy, q);
        sum += x[6*q - 4] * pow(p2.healthcare, q);
        sum += x[6*q - 3] * pow(p2.landlocked, q);
        sum += x[6*q - 2] * pow(p2.temperature, q);
        sum += x[6*q - 1] * pow(log10(p2.area), q);
    }
    return pow(10, sum);
}

void variance_explained(VectorXd& x, std::vector<double>& predicted_population)
{
    int row_am = out_p.size();
    double y_bar = 0;
    for (int i = 0; i < row_am; i++)
    {
        y_bar += log10(out_p[i].population);
    }
    y_bar /= row_am;

    double num = 0;
    double den = 0;
    for (int i = 0; i < row_am; i++)
    {
        num += pow(log10(out_p[i].population) - log10(predicted_population[i]),2);
        den += pow(log10(out_p[i].population) - y_bar, 2);
    }

    cout << "variance explained : " << 1 - num / den << '\n';
}

VectorXd matrix_calculate(int row_am, int total_population_weight)
{
    MatrixXd A(row_am + total_population_weight, k * 6 + 1);
    VectorXd B(row_am + total_population_weight);
    for (int i = 0; i < row_am; i++)
    {
        B(i) = log10(out_p[i].population);

        for (int q = 1; q <= k; q++)
        {
            A(i, q * 6 - 6) = pow(log10(out_p[i].gdp_capita), q);
            A(i, q * 6 - 5) = pow(out_p[i].life_expectancy, q);
            A(i, q * 6 - 4) = pow(out_p[i].healthcare, q);
            A(i, q * 6 - 3) = pow(out_p[i].landlocked, q);
            A(i, q * 6 - 2) = pow(out_p[i].temperature, q);
            A(i, q * 6 - 1) = pow(log10(out_p[i].area), q);
        }
        A(i, k * 6) = 1;
    }

    return A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
}

void global_population(VectorXd& x)
{
    long long sum = 0;
    long long delta = 0;

    for (int p = 0; p < out_p.size(); p++)
    {
        sum += eval_f(out_p[p], x);
        delta += out_p[p].population - eval_f(out_p[p], x);
    }
    cout << "Total estimated world population: " << std::fixed << sum << '\n';
}
int main() 
{    
    cout.imbue(std::locale("en_US.UTF-8"));
    load_data();
    int row_am = out_p.size();
    VectorXd x = matrix_calculate(row_am, 1);
    cout << x;

    vector<double> predicted_population;
    for (int i = 0; i < row_am; i++)
    {
        predicted_population.push_back(eval_f(out_p[i], x));
    }

    global_population(x);
    variance_explained(x, predicted_population);


    while (true)
    {
        string count;
        std::cin >> count;
        int count_ind = country_index_by_id[count];
        
        cout << out_p[count_ind].country_name << "\n estimated pop: "
            << std::fixed << (int)predicted_population[count_ind] << "\n true pop: "
            << out_p[count_ind].population << "\n delta pop: " << (int)abs(out_p[count_ind].population - predicted_population[count_ind])
            << "\n \n";
    }

    /*
        -0.371136       gdp_capita
        0.03633         life_expectancy
        -0.0108641      healthcare
        -0.0442496      landlocked
        0.00761164      temperature
        0.719452        area
        2.12836         constant
    */
}
