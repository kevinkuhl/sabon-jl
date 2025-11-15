function run_analysis(model, gridmat, grid, train_in, train_out, val_in, val_out, test_in, test_out, config, output_dir)
    mkpath(output_dir)
    
    # Move to CPU for analysis
    model_cpu = cpu(model)
    gridmat_cpu = cpu(gridmat)
    
    # Get basis
    basis = model_cpu.basis(gridmat_cpu)
    
    # 1. Compute principal angles
    pa = compute_principal_angles_with_cat_map(basis, grid, gridmat_cpu)
    
    # 2. Compute spectrum
    G = basis * basis' / size(basis, 2)
    spectrum, eigvecs = compute_spectrum(model_cpu.linear_map, basis)
    
    # 3. Compute final metrics
    train_loss = Flux.Losses.mse(model_cpu(train_in, gridmat_cpu), train_out)
    val_loss = Flux.Losses.mse(model_cpu(val_in, gridmat_cpu), val_out)
    test_loss = Flux.Losses.mse(model_cpu(test_in, gridmat_cpu), test_out)
    
    train_mre = mean(abs.((model_cpu(train_in, gridmat_cpu) .- train_out) ./ train_out))
    val_mre = mean(abs.((model_cpu(val_in, gridmat_cpu) .- val_out) ./ val_out))
    test_mre = mean(abs.((model_cpu(test_in, gridmat_cpu) .- test_out) ./ test_out))
    
    # Save metrics
    JLD2.jldsave(
        joinpath(output_dir, "metrics.jld2");
        principal_angles = pa,
        spectrum = spectrum,
        train_loss = train_loss,
        val_loss = val_loss,
        test_loss = test_loss,
        train_mre = train_mre,
        val_mre = val_mre,
        test_mre = test_mre
    )
    
    # 4. Generate visualizations
    # (Add your visualization code from the Julia analysis file)
    
    @info """
    Analysis Results:
    - Train Loss: $train_loss, MRE: $train_mre
    - Val Loss: $val_loss, MRE: $val_mre  
    - Test Loss: $test_loss, MRE: $test_mre
    - Principal Angles: min=$(minimum(pa))°, max=$(maximum(pa))°
    """
end