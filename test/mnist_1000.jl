using JLD2

function test_solver(solver, test_selected)

    # index of all violated cases in the first 1000 images.
    violation_idx = [25, 142, 179, 183, 283, 285, 387, 392, 612, 647, 737]
    # index of selected holding cases in the first 1000 images.
    satisfied_idx = [10, 100, 200, 300, 400, 500 ,600, 700, 800, 900, 999, 213, 391, 660, 911]

    if test_selected
        test_idx = [violation_idx; satisfied_idx]
    else
        test_idx = 1:1000
    end

    @load "MNIST_1000.jld2" train_x train_y mnist_net

    for i in test_idx
        input_center = reshape(train_x[:,:,i], 28*28)
        label = train_y[i]
        A = zeros(Float64, 10, 10)
        A[diagind(A)] .= 1
        A[:, label+1] = A[:, label+1] .- 1
        A = A[1:end .!= label+1,:]

        b = zeros(Float64, 9)

        Y = HPolytope(A, b)
        pred = argmax(NeuralVerification.compute_output(mnist_net, input_center))-1

        if pred != label
            continue
        end

        epsilon = 1.0/255.0
        upper = input_center .+ epsilon
        lower = input_center .- epsilon
        clamp!(upper, 0.0, 1.0)
        clamp!(lower, 0.0, 1.0)
        X = Hyperrectangle(low=lower, high=upper)

        problem_mnist = Problem(mnist_net, X, Y)

        result =  solve(solver, problem_mnist)

        if result.status == :violated
            noisy = argmax(NeuralVerification.compute_output(mnist_net, result.counter_example))-1
            @test i ∈ violation_idx && noisy != pred
        elseif result.status == :holds
            @test i ∉ violation_idx
        end
    end
end

let
    # if test_selected = true, only test selected index. Otherwise, test all 1000 images.
    test_selected = true
    testname =  join(["MNIST ", (test_selected ? "selected" : "1000")])

    # NOTE the number of tests maybe different for different solvers.
    # Because the test cases depends on the result.
    @time @testset "$testname, ReluVal" begin
        test_solver(ReluVal(max_iter = 10), test_selected)
    end
    @time @testset "$testname, Neurify" begin
        test_solver(Neurify(max_iter = 10), test_selected)
    end

end