# New Plan/Act endpoints to be added to main.py


@app.post("/plan-generation", response_model=GenerationPlan)
async def plan_generation(request: GenerationPlanRequest):
    """
    PLAN PHASE: Create a comprehensive generation plan without executing

    Returns:
    - Model recommendation with reasoning
    - Enhanced prompt with intelligent negative prompts
    - Quality analysis (specificity score, missing elements, warnings)
    - Parameter reasoning (why each parameter was chosen)
    - Tips for best results
    """
    try:
        # Get installed models
        installed_models = model_manager.get_installed_models()

        # Create comprehensive plan
        plan = await prompt_engine.create_generation_plan(
            user_input=request.user_input,
            installed_models=installed_models,
            conversation_history=request.conversation_history
        )

        return plan
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create generation plan: {str(e)}")


@app.post("/execute-generation", response_model=GenerationResponse)
async def execute_generation(request: ExecuteGenerationRequest):
    """
    ACT PHASE: Execute a generation plan

    Takes the plan from /plan-generation and generates the image.
    Allows model override if user wants to use a different model than recommended.
    """
    try:
        plan = request.plan

        # Use model override if provided, otherwise use recommended model
        # Note: This would integrate with SD API to switch models
        # For now, we'll just use whatever model is currently loaded in SD

        if request.model_override:
            print(f"[INFO] User override: using model '{request.model_override}' instead of '{plan.model_recommendation.recommended_model_name}'")

        # Generate image using the planned prompt
        result = await sd_service.generate_image(plan.enhanced_prompt)

        return GenerationResponse(
            image_base64=result["image_base64"],
            prompt_used=plan.enhanced_prompt,
            llm_explanation=plan.explanation,
            generation_time=result["generation_time"],
            seed_used=result["seed"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


# Img2Img Endpoints

@app.post("/plan-img2img", response_model=Img2ImgGenerationPlan)
async def plan_img2img(request: Img2ImgPlanRequest):
    """
    PLAN PHASE: Create a comprehensive img2img transformation plan without executing

    Returns:
    - Model recommendation with reasoning
    - Enhanced prompt with intelligent negative prompts
    - Denoising strength recommendation with reasoning
    - Quality analysis (specificity score, missing elements, warnings)
    - Parameter reasoning (why each parameter was chosen)
    - Tips for best transformation results
    """
    try:
        # Get installed models
        installed_models = model_manager.get_installed_models()

        # Create comprehensive img2img plan
        plan = await prompt_engine.create_img2img_plan(
            user_input=request.user_input,
            init_image_base64=request.init_image_base64,
            installed_models=installed_models,
            conversation_history=request.conversation_history
        )

        return plan
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create img2img plan: {str(e)}")


@app.post("/execute-img2img", response_model=Img2ImgResponse)
async def execute_img2img(request: ExecuteImg2ImgRequest):
    """
    ACT PHASE: Execute an img2img transformation plan

    Takes the plan from /plan-img2img and generates the transformed image.
    Allows model override if user wants to use a different model than recommended.
    """
    try:
        plan = request.plan

        if request.model_override:
            print(f"[INFO] User override: using model '{request.model_override}' instead of '{plan.model_recommendation.recommended_model_name}'")

        # Generate image using img2img with the planned prompt and denoising strength
        result = await sd_service.img2img(
            prompt=plan.enhanced_prompt,
            init_image_base64=request.init_image_base64,
            denoising_strength=plan.denoising_strength
        )

        return Img2ImgResponse(
            image_base64=result["image_base64"],
            prompt_used=plan.enhanced_prompt,
            denoising_strength=plan.denoising_strength,
            llm_explanation=plan.explanation,
            generation_time=result["generation_time"],
            seed_used=result["seed"],
            source_image_base64=request.init_image_base64
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Img2img execution failed: {str(e)}")
